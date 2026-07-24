#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace kda {

static constexpr int conv1d_tile_size = 8;

template <typename T, typename CacheT, int Width, int TileT>
struct causal_conv1d_tiled_kernel {
  static constexpr int sub_group_size = 16;
  static constexpr int elements_per_item = 4;
  static constexpr int work_group_size = 64;
  static constexpr int features_per_work_group =
      work_group_size * elements_per_item;
  static constexpr int metadata_ints = 4;
  static constexpr int metadata_bytes = metadata_ints * sizeof(int);
  static constexpr int input_elements =
      (TileT + Width - 1) * features_per_work_group;

  causal_conv1d_tiled_kernel(
      T* q,
      T* k,
      T* v,
      const T* q_proj,
      const T* k_proj,
      const T* v_proj,
      const float* q_weight,
      const float* k_weight,
      const float* v_weight,
      const CacheT* conv_state,
      int64_t conv_state_stride_0,
      int64_t conv_state_dim_stride,
      int64_t conv_state_time_stride,
      CacheT* conv_state_tmp,
      const int* query_start_loc,
      const int* token_indx,
      const int* state_indices,
      const bool* has_initial_state,
      int batch_size,
      int hidden_dim,
      char* local_memory)
      : q(q),
        k(k),
        v(v),
        q_proj(q_proj),
        k_proj(k_proj),
        v_proj(v_proj),
        q_weight(q_weight),
        k_weight(k_weight),
        v_weight(v_weight),
        conv_state(conv_state),
        conv_state_stride_0(conv_state_stride_0),
        conv_state_dim_stride(conv_state_dim_stride),
        conv_state_time_stride(conv_state_time_stride),
        conv_state_tmp(conv_state_tmp),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        state_indices(state_indices),
        has_initial_state(has_initial_state),
        batch_size(batch_size),
        hidden_dim(hidden_dim),
        local_memory(local_memory) {}

  static int get_num_feature_chunks(int hidden_dim) {
    return (hidden_dim + features_per_work_group - 1) / features_per_work_group;
  }

  static sycl::nd_range<2> get_nd_range(int num_tiles, int hidden_dim) {
    const int num_feature_chunks = get_num_feature_chunks(hidden_dim);
    return sycl::nd_range<2>(
        sycl::range<2>(num_tiles * num_feature_chunks, 3 * work_group_size),
        sycl::range<2>(1, work_group_size));
  }

  static constexpr int get_local_memory_bytes() {
    return metadata_bytes + input_elements * sizeof(CacheT);
  }

  int lookup(int token) const {
    return token_indx == nullptr ? token : token_indx[token];
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int stream = item.get_group(1);
    const int local_id = item.get_local_id(1);
    const int num_feature_chunks = get_num_feature_chunks(hidden_dim);
    const int combined_group = item.get_group(0);
    const int tile_id = combined_group / num_feature_chunks;
    const int feature_chunk = combined_group % num_feature_chunks;

    int* metadata = reinterpret_cast<int*>(local_memory);
    CacheT* input_tile =
        reinterpret_cast<CacheT*>(local_memory + metadata_bytes);

    if (local_id == 0) {
      int batch_id = -1;
      int tile_start = 0;
      int tiles_before = 0;
      for (int batch = 0; batch < batch_size; ++batch) {
        const int seq_start = query_start_loc[batch];
        const int seq_end = query_start_loc[batch + 1];
        const int sequence_tiles = (seq_end - seq_start + TileT - 1) / TileT;
        if (tile_id < tiles_before + sequence_tiles) {
          batch_id = batch;
          tile_start = (tile_id - tiles_before) * TileT;
          metadata[0] = batch_id;
          metadata[1] = tile_start;
          metadata[2] = seq_start;
          metadata[3] = seq_end;
          break;
        }
        tiles_before += sequence_tiles;
      }
      if (batch_id < 0) {
        metadata[0] = -1;
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    const int batch_id = metadata[0];
    if (batch_id < 0) {
      return;
    }
    const int tile_start = metadata[1];
    const int seq_start = metadata[2];
    const int seq_end = metadata[3];
    const int sequence_length = seq_end - seq_start;
    const int tile_tokens = sycl::min(TileT, sequence_length - tile_start);
    const int state_id = state_indices[batch_id];
    if (state_id == pad_slot_id) {
      return;
    }

    const int channel_start =
        feature_chunk * features_per_work_group + local_id * elements_per_item;
    const bool channel_valid = channel_start < hidden_dim;
    const int combined_channel = stream * hidden_dim + channel_start;
    const bool load_initial_state =
        has_initial_state == nullptr || has_initial_state[batch_id];
    const T* input = stream == 0 ? q_proj : (stream == 1 ? k_proj : v_proj);
    T* output = stream == 0 ? q : (stream == 1 ? k : v);
    const float* weight =
        stream == 0 ? q_weight : (stream == 1 ? k_weight : v_weight);

    if (channel_valid) {
      for (int slot = 0; slot < TileT + Width - 1; ++slot) {
        const int token_in_sequence = tile_start + slot - (Width - 1);
#pragma unroll
        for (int element = 0; element < elements_per_item; ++element) {
          const int channel = channel_start + element;
          CacheT value = static_cast<CacheT>(0.0f);
          if (token_in_sequence < 0) {
            const int state_time = Width - 1 + token_in_sequence;
            if (load_initial_state) {
              value = conv_state
                  [static_cast<int64_t>(state_id) * conv_state_stride_0 +
                   static_cast<int64_t>(stream * hidden_dim + channel) *
                       conv_state_dim_stride +
                   static_cast<int64_t>(state_time) * conv_state_time_stride];
            }
          } else if (token_in_sequence < sequence_length) {
            const int global_token = lookup(seq_start + token_in_sequence);
            value = static_cast<CacheT>(
                input
                    [static_cast<int64_t>(global_token) * hidden_dim +
                     channel]);
          }
          input_tile
              [slot * features_per_work_group + local_id * elements_per_item +
               element] = value;
        }
      }
    }
    item.barrier(sycl::access::fence_space::local_space);

    if (!channel_valid) {
      return;
    }

    float local_weight[Width * elements_per_item];
#pragma unroll
    for (int kernel = 0; kernel < Width; ++kernel) {
#pragma unroll
      for (int element = 0; element < elements_per_item; ++element) {
        local_weight[kernel * elements_per_item + element] = weight
            [static_cast<int64_t>(channel_start + element) * Width + kernel];
      }
    }

    for (int token = 0; token < tile_tokens; ++token) {
      float result[elements_per_item] = {};
#pragma unroll
      for (int kernel = 0; kernel < Width; ++kernel) {
#pragma unroll
        for (int element = 0; element < elements_per_item; ++element) {
          result[element] +=
              static_cast<float>(
                  input_tile
                      [(token + kernel) * features_per_work_group +
                       local_id * elements_per_item + element]) *
              local_weight[kernel * elements_per_item + element];
        }
      }

      const int global_token = lookup(seq_start + tile_start + token);
#pragma unroll
      for (int element = 0; element < elements_per_item; ++element) {
        output
            [static_cast<int64_t>(global_token) * hidden_dim + channel_start +
             element] = static_cast<T>(silu(result[element]));
      }
    }

    if (tile_start + TileT >= sequence_length) {
      const int last_slot = tile_tokens - 1 + Width - 1;
#pragma unroll
      for (int state_time = 0; state_time < Width - 1; ++state_time) {
#pragma unroll
        for (int element = 0; element < elements_per_item; ++element) {
          conv_state_tmp
              [(static_cast<int64_t>(batch_id) * 3 * hidden_dim +
                combined_channel + element) *
                   (Width - 1) +
               state_time] = input_tile
                  [(last_slot - (Width - 2) + state_time) *
                       features_per_work_group +
                   local_id * elements_per_item + element];
        }
      }
    }
  }

 private:
  T* q;
  T* k;
  T* v;
  const T* q_proj;
  const T* k_proj;
  const T* v_proj;
  const float* q_weight;
  const float* k_weight;
  const float* v_weight;
  const CacheT* conv_state;
  int64_t conv_state_stride_0;
  int64_t conv_state_dim_stride;
  int64_t conv_state_time_stride;
  CacheT* conv_state_tmp;
  const int* query_start_loc;
  const int* token_indx;
  const int* state_indices;
  const bool* has_initial_state;
  int batch_size;
  int hidden_dim;
  char* local_memory;
};

template <typename CacheT, int Width>
struct update_conv_state_kernel {
  update_conv_state_kernel(
      CacheT* conv_state,
      int64_t conv_state_stride_0,
      int64_t conv_state_dim_stride,
      int64_t conv_state_time_stride,
      const CacheT* conv_state_tmp,
      const int* state_indices,
      int batch_size,
      int hidden_dim)
      : conv_state(conv_state),
        conv_state_stride_0(conv_state_stride_0),
        conv_state_dim_stride(conv_state_dim_stride),
        conv_state_time_stride(conv_state_time_stride),
        conv_state_tmp(conv_state_tmp),
        state_indices(state_indices),
        batch_size(batch_size),
        hidden_dim(hidden_dim) {}

  static sycl::nd_range<2> get_nd_range(int batch_size, int hidden_dim) {
    const int combined_dim = 3 * hidden_dim;
    const int rounded_dim = (combined_dim + work_group_size - 1) /
                            work_group_size * work_group_size;
    return sycl::nd_range<2>(
        sycl::range<2>(batch_size, rounded_dim),
        sycl::range<2>(1, work_group_size));
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<2> item) const {
    const int batch_id = item.get_group(0);
    const int combined_channel = item.get_global_id(1);
    if (batch_id >= batch_size || combined_channel >= 3 * hidden_dim) {
      return;
    }
    const int state_id = state_indices[batch_id];
    if (state_id == pad_slot_id) {
      return;
    }

#pragma unroll
    for (int state_time = 0; state_time < Width - 1; ++state_time) {
      conv_state
          [static_cast<int64_t>(state_id) * conv_state_stride_0 +
           static_cast<int64_t>(combined_channel) * conv_state_dim_stride +
           static_cast<int64_t>(state_time) * conv_state_time_stride] =
              conv_state_tmp
                  [(static_cast<int64_t>(batch_id) * 3 * hidden_dim +
                    combined_channel) *
                       (Width - 1) +
                   state_time];
    }
  }

 private:
  static constexpr int sub_group_size = 32;
  static constexpr int work_group_size = 256;

  CacheT* conv_state;
  int64_t conv_state_stride_0;
  int64_t conv_state_dim_stride;
  int64_t conv_state_time_stride;
  const CacheT* conv_state_tmp;
  const int* state_indices;
  int batch_size;
  int hidden_dim;
};

template <typename T, typename CacheT, int Width>
void launch_causal_conv1d_tiled(
    sycl::queue& queue,
    T* q,
    T* k,
    T* v,
    const T* q_proj,
    const T* k_proj,
    const T* v_proj,
    const float* q_weight,
    const float* k_weight,
    const float* v_weight,
    CacheT* conv_state,
    int64_t conv_state_stride_0,
    int64_t conv_state_dim_stride,
    int64_t conv_state_time_stride,
    CacheT* conv_state_tmp,
    const int* query_start_loc,
    const int* token_indx,
    const int* state_indices,
    const bool* has_initial_state,
    int batch_size,
    int num_tokens,
    int hidden_dim) {
  constexpr int TileT = conv1d_tile_size;
  using MainKernel = causal_conv1d_tiled_kernel<T, CacheT, Width, TileT>;
  const int num_tiles = (num_tokens + TileT - 1) / TileT + batch_size;
  const auto main_range = MainKernel::get_nd_range(num_tiles, hidden_dim);
  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<char, 1> local_memory(
        sycl::range<1>(MainKernel::get_local_memory_bytes()), cgh);
    cgh.parallel_for(main_range, [=](sycl::nd_item<2> item) {
      char* local_memory_ptr =
          local_memory.template get_multi_ptr<sycl::access::decorated::no>()
              .get_raw();
      MainKernel task(
          q,
          k,
          v,
          q_proj,
          k_proj,
          v_proj,
          q_weight,
          k_weight,
          v_weight,
          conv_state,
          conv_state_stride_0,
          conv_state_dim_stride,
          conv_state_time_stride,
          conv_state_tmp,
          query_start_loc,
          token_indx,
          state_indices,
          has_initial_state,
          batch_size,
          hidden_dim,
          local_memory_ptr);
      task(item);
    });
  });

  using UpdateKernel = update_conv_state_kernel<CacheT, Width>;
  const auto update_range = UpdateKernel::get_nd_range(batch_size, hidden_dim);
  queue.submit([&](sycl::handler& cgh) {
    UpdateKernel task(
        conv_state,
        conv_state_stride_0,
        conv_state_dim_stride,
        conv_state_time_stride,
        conv_state_tmp,
        state_indices,
        batch_size,
        hidden_dim);
    cgh.parallel_for(update_range, task);
  });
}

}  // namespace kda
