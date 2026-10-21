/**
 * part of LukasNalbach/compact-texts
 *
 * MIT License
 *
 * Copyright (c) Lukas Nalbach
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef COMPACT_TEXTS_SPLIT_TEXT_HPP
#define COMPACT_TEXTS_SPLIT_TEXT_HPP

#include <omp.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>

#if defined(__AVX512VBMI__) && defined(__AVX512BW__)
#include <immintrin.h>
#define COMPACT_TEXTS_SPLIT_TEXT_VBMI 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define COMPACT_TEXTS_SPLIT_TEXT_AVX2 1
#endif

#ifndef COMPACT_TEXTS_SPLIT_TEXT_NOINLINE
#if defined(__GNUC__)
#define COMPACT_TEXTS_SPLIT_TEXT_NOINLINE __attribute__((noinline))
#else
#define COMPACT_TEXTS_SPLIT_TEXT_NOINLINE
#endif
#endif

#include <misc/memory.hpp>

#include <compact_texts/common.hpp>
#include <compact_texts/packed_text.hpp>

namespace compact_texts {

template <typename t_symbol = uint8_t>
class split_text;

template <>
class split_text<uint8_t> {
 public:
  using char_type = char;
  using symbol_type = uint8_t;
  using histogram_t = packed_text<>::histogram_t;

  static constexpr bool is_byte_text = true;
  static constexpr uint64_t payload_penalty = 2;
  static constexpr uint64_t simd_symbols = 64;
  static constexpr uint64_t batches_per_block = 4;
  static constexpr uint64_t block_symbols = batches_per_block * simd_symbols;
  static constexpr uint64_t chunk_symbols = block_symbols;
  static constexpr uint64_t super_symbols = uint64_t{1} << 20;
  static constexpr uint64_t blocks_per_super = super_symbols / block_symbols;
  static constexpr uint64_t offset_bytes = 3;
  static constexpr uint64_t max_payload = 8;
  static constexpr uint64_t max_split_width = 6;
  static constexpr uint64_t padding_bytes = 128;
  static constexpr uint64_t medium_symbols = 192;
  static constexpr uint64_t window_bits = 56;
  static constexpr uint64_t window_mask = (uint64_t{1} << window_bits) - 1;
  static constexpr int64_t predicted_lines = 3;
  static constexpr uint32_t sparse_x256 = 8;
  static constexpr uint64_t eager_denominator = 5;

  static_assert(super_symbols * max_payload <= uint64_t{1} << (8 * offset_bytes));

  class cursor {
   public:
    cursor() = default;

    cursor(const split_text* text, uint64_t i)
        : m_text(text), m_batch(i / simd_symbols * simd_symbols), m_at(uint32_t(i - m_batch)) {
      m_bit = text->m_has_payloads ? text->payload_bit(i) : 0;
      text->template decode_batch<false>(m_batch, m_at, simd_symbols, m_bit, m_buffer);
    }

    uint8_t next() {
      if (m_at == simd_symbols) [[unlikely]] {
        m_batch += simd_symbols;
        m_at = 0;
        m_text->template decode_batch<false>(m_batch, 0, simd_symbols, m_bit, m_buffer);
      }

      return m_buffer[m_at++];
    }

   private:
    alignas(64) uint8_t m_buffer[simd_symbols];
    const split_text* m_text = nullptr;
    uint64_t m_batch = 0;
    uint64_t m_bit = 0;
    uint32_t m_at = 0;
  };

  split_text() = default;

  split_text(const char* data, uint64_t size, int threads = omp_get_max_threads())
      : split_text(packed_text<>::count_chars(data, size, threads), size) {
    pack(data, size, 0, threads);
    finish(threads);
  }

  split_text(const histogram_t& histogram, uint64_t size, uint64_t penalty = payload_penalty)
      : m_size(size), m_penalty(penalty) {
    m_data = std::make_shared<data_t>();
    data_t& d = *m_data;
    std::array<uint64_t, 256> freq{};
    std::array<uint8_t, 256> order{};
    uint64_t sigma = 0;

    for (uint16_t c = 0; c < 256; ++c) {
      if (histogram[c] == 0) continue;
      d.to_symbol[c] = uint8_t(sigma);
      d.to_char[sigma] = uint8_t(c);
      freq[sigma] = histogram[c];
      order[sigma] = uint8_t(sigma);
      ++sigma;
    }

    m_sigma = uint16_t(std::max<uint64_t>(sigma, 1));
    std::stable_sort(order.begin(), order.begin() + sigma, [&](uint8_t a, uint8_t b) { return freq[a] > freq[b]; });
    std::vector<uint64_t> sorted(sigma);
    for (uint64_t r = 0; r < sigma; ++r) sorted[r] = freq[order[r]];
    std::vector<uint8_t> lengths;
    m_width = choose_code(sorted, size, penalty, lengths);
    const uint64_t classes = uint64_t{1} << m_width;
    d.symbol_of.assign(classes << max_payload, 0);
    d.char_of.assign(classes << max_payload, 0);
    uint64_t next = 0;
    uint64_t payload_chars = 0;
    m_direct_classes = 0;

    for (uint64_t c = 0; c < lengths.size(); ++c) {
      const uint64_t slots = uint64_t{1} << lengths[c];
      d.len[c] = lengths[c];
      d.mask[c] = uint8_t(slots - 1);
      m_has_payloads = m_has_payloads || lengths[c] != 0;
      if (lengths[c] == 0) m_direct_classes = uint16_t(c + 1);

      for (uint64_t p = 0; p < slots && next < sigma; ++p, ++next) {
        const uint8_t symbol = order[next];
        d.class_of[symbol] = uint8_t(c);
        d.payload_of[symbol] = uint8_t(p);
        d.symbol_of[(c << max_payload) | p] = symbol;
        d.char_of[(c << max_payload) | p] = d.to_char[symbol];
        if (lengths[c] != 0) payload_chars += freq[symbol];
      }
    }

    m_eager = size != 0 && payload_chars * eager_denominator >= size;

    for (uint16_t c = 0; c < 256; ++c) {
      if (histogram[c] == 0) continue;
      const uint8_t symbol = d.to_symbol[c];
      d.class_of_char[c] = d.class_of[symbol];
      d.payload_of_char[c] = d.payload_of[symbol];
      d.len_of_char[c] = d.len[d.class_of[symbol]];
    }

    for (uint64_t k = 0; k < 64; ++k) {
#if defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
      const uint64_t bit = ((k >> 1) & 7) * m_width;
      const uint64_t scale = uint64_t{256} >> (bit & 7);
      d.spread[k] = uint8_t((bit >> 3) + (k & 1));
      d.shifts[k] = uint8_t((k & 1) != 0 ? scale >> 8 : scale);
#else
      d.spread[k] = uint8_t((k >> 3) * m_width + (k & 7));
      d.shifts[k] = uint8_t((k & 7) * m_width);
#endif
      d.direct_symbol[k] = k < classes ? d.symbol_of[k << max_payload] : 0;
      d.direct_char[k] = k < classes ? d.char_of[k << max_payload] : 0;
      d.bit_to_symbol[k] = uint8_t(k / m_width);
    }

    m_class_mask = uint8_t(classes - 1);
    m_short = uint8_t(std::min<uint64_t>(window_bits / m_width, simd_symbols - 1));
    m_short_mask = (uint64_t{1} << (m_short * m_width)) - 1;

    for (uint64_t field = 0; field * m_width + m_width <= 64; ++field) {
      m_field_one |= uint64_t{1} << (field * m_width);
      m_field_high |= uint64_t{1} << (field * m_width + m_width - 1);
    }

    m_field_low = m_field_high - m_field_one;
    const uint64_t direct = std::min<uint64_t>(m_direct_classes, classes - 1);
    m_direct_high = ((direct >> (m_width - 1)) & 1) != 0;
    m_direct_low = m_field_one * (direct & ((uint64_t{1} << (m_width - 1)) - 1));
    histogram_t class_histogram{};
    for (uint64_t c = 0; c < classes; ++c) class_histogram[c] = 1;
    m_codes = packed_text<>(class_histogram, size);

    if (m_has_payloads) {
      d.abs.assign(size / super_symbols + 2, 0);
      d.rel.assign(offset_bytes * (size / block_symbols + 2) + 8, 0);
    }

    d.payload.assign(padding_bytes, 0);
    cache_pointers();
  }

  void pack(const char* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) {
    const uint8_t* class_of_char = m_data->class_of_char.data();
    m_codes.pack_symbols(length, at, [class_of_char, data](uint64_t i) { return uint64_t(class_of_char[uint8_t(data[i])]); }, threads);
    if (!m_has_payloads || length == 0) return;
    const uint8_t* payload_of_char = m_data->payload_of_char.data();
    const uint8_t* len_of_char = m_data->len_of_char.data();
    const uint64_t blocks = (length + block_symbols - 1) / block_symbols;
    const uint64_t parts = std::min<uint64_t>(blocks, std::max(threads, 1));

#pragma omp parallel for num_threads(threads)
    for (uint64_t part = 0; part < parts; ++part) {
      const uint64_t first_block = blocks * part / parts;
      const uint64_t last_block = blocks * (part + 1) / parts;
      const uint64_t beg = first_block * block_symbols;
      const uint64_t end = std::min(length, last_block * block_symbols);
      piece_t piece;
      piece.at = at + beg;
      piece.block_bits.assign(last_block - first_block, 0);
      uint64_t total = 0;

      for (uint64_t b = first_block; b < last_block; ++b) {
        const uint64_t block_end = std::min(end, (b + 1) * block_symbols);
        uint64_t bits = 0;
        for (uint64_t i = b * block_symbols; i < block_end; ++i) bits += len_of_char[uint8_t(data[i])];
        piece.block_bits[b - first_block] = uint16_t(bits);
        total += bits;
      }

      piece.bits = total;
      resize_no_init(piece.words, (total + 63) / 64 + 1);
      uint64_t word = 0;
      uint64_t acc = 0;
      uint64_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint8_t c = uint8_t(data[i]);
        const uint64_t value = payload_of_char[c];
        const uint64_t len = len_of_char[c];
        acc |= value << acc_bits;
        acc_bits += len;

        if (acc_bits >= 64) {
          piece.words[word++] = acc;
          acc_bits -= 64;
          acc = acc_bits == 0 ? 0 : value >> (len - acc_bits);
        }
      }

      if (acc_bits != 0) piece.words[word++] = acc;
      piece.words.resize(word);
      std::lock_guard<std::mutex> lock(m_data->mutex);
      m_data->pieces.push_back(std::move(piece));
    }
  }

  void finish(int threads = omp_get_max_threads()) {
    data_t& d = *m_data;

    if (!m_has_payloads) {
      d.pieces = decltype(d.pieces)();
      cache_pointers();
      return;
    }

    std::sort(d.pieces.begin(), d.pieces.end(), [](const piece_t& a, const piece_t& b) { return a.at < b.at; });
    const uint64_t pieces = d.pieces.size();
    std::vector<uint64_t> offset(pieces + 1, 0);
    for (uint64_t p = 0; p < pieces; ++p) offset[p + 1] = offset[p] + d.pieces[p].bits;
    const uint64_t total = offset[pieces];
    m_payload_bits = total;
    m_avg_x256 = uint32_t(m_size == 0 ? 0 : (total * 256 + m_size / 2) / m_size);
    resize_no_init(d.payload, (total + 63) / 64 * 8 + padding_bytes);
    advise_huge_pages(d.payload.data(), d.payload.size());
    parallel_memset(d.payload.data(), 0, d.payload.size(), threads);

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        if (block % blocks_per_super == 0) d.abs[block / blocks_per_super] = cum;
        cum += piece.block_bits[k];
      }
    }

    const uint64_t end_block = (m_size + block_symbols - 1) / block_symbols;
    if (end_block % blocks_per_super == 0) d.abs[end_block / blocks_per_super] = total;

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        write_offset(d.rel.data(), block, cum - d.abs[block / blocks_per_super]);
        cum += piece.block_bits[k];
      }
    }

    write_offset(d.rel.data(), end_block, total - d.abs[end_block / blocks_per_super]);
    uint64_t* out = reinterpret_cast<uint64_t*>(d.payload.data());

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      if (piece.bits == 0) continue;
      const uint64_t first = offset[p] >> 6;
      const uint64_t last = (offset[p] + piece.bits - 1) >> 6;
      const uint64_t shift = offset[p] & 63;
      const uint64_t words = (piece.bits + 63) / 64;
      const uint64_t tail = piece.bits & 63;
      uint64_t carry = 0;

      for (uint64_t w = first; w <= last; ++w) {
        const uint64_t k = w - first;
        uint64_t x = k < words ? piece.words[k] : 0;
        if (k + 1 == words && tail != 0) x &= (uint64_t{1} << tail) - 1;
        const uint64_t value = (x << shift) | carry;
        carry = shift == 0 ? 0 : x >> (64 - shift);
        if (w == first || w == last) std::atomic_ref<uint64_t>(out[w]).fetch_or(value, std::memory_order_relaxed);
        else out[w] = value;
      }
    }

    d.pieces = decltype(d.pieces)();
    cache_pointers();
  }

  uint64_t size() const { return m_size; }
  uint64_t sigma() const { return m_sigma; }
  uint8_t width() const { return m_width; }
  uint64_t penalty() const { return m_penalty; }
  bool has_payloads() const { return m_has_payloads; }
  uint64_t payload_bits() const { return m_payload_bits; }
  const uint8_t* class_lengths() const { return m_len; }
  const uint8_t* class_of_chars() const { return m_data->class_of_char.data(); }

  uint64_t size_in_bytes() const {
    uint64_t bytes = sizeof(*this) + m_codes.size_in_bytes();
    if (m_data) {
      bytes += sizeof(data_t) + m_data->payload.size() + m_data->abs.size() * 8 + m_data->rel.size() +
               m_data->symbol_of.size() + m_data->char_of.size();
    }
    return bytes;
  }

  static uint64_t size_in_bytes_for(const histogram_t& histogram, uint64_t size, uint64_t penalty = payload_penalty) {
    std::vector<uint64_t> sorted;
    for (uint16_t c = 0; c < 256; ++c) {
      if (histogram[c] != 0) sorted.push_back(histogram[c]);
    }

    std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());
    std::vector<uint8_t> lengths;
    const uint8_t width = choose_code(sorted, size, penalty, lengths);
    uint64_t payload_bits = 0;
    uint64_t next = 0;
    bool any_payload = false;

    for (uint8_t l : lengths) {
      any_payload = any_payload || l != 0;
      for (uint64_t p = 0; p < (uint64_t{1} << l) && next < sorted.size(); ++p, ++next) payload_bits += sorted[next] * l;
    }

    uint64_t bytes = sizeof(split_text) + sizeof(packed_text<>) + (size * width + 63) / 64 * 8 + packed_text<>::padding_bytes +
                     sizeof(data_t) + 2 * ((uint64_t{1} << width) << max_payload);

    if (!any_payload) return bytes + padding_bytes;
    return bytes + (payload_bits + 63) / 64 * 8 + padding_bytes + (size / super_symbols + 2) * 8 +
           offset_bytes * (size / block_symbols + 2) + 8;
  }

  uint8_t operator[](uint64_t i) const {
    const uint8_t c = m_codes[i];
    if (m_len[c] == 0) return m_symbol_of[uint64_t(c) << max_payload];
    return payload_symbol<false>(i, c);
  }

  char_type char_at(uint64_t i) const {
    const uint8_t c = m_codes[i];
    if (m_len[c] == 0) return char_type(m_char_of[uint64_t(c) << max_payload]);
    return char_type(payload_symbol<true>(i, c));
  }

  bool less_char(uint64_t i, uint64_t j) const { return (*this)[i] < (*this)[j]; }
  uint8_t to_char(uint8_t symbol) const { return m_to_char[symbol]; }
  uint8_t to_symbol(uint8_t c) const { return m_to_symbol[c]; }
  cursor cursor_at(uint64_t i) const { return cursor(this, i); }

  void extract(uint64_t i, uint64_t len, uint8_t* __restrict out) const {
    if (len == 0) return;
    if (m_has_payloads) prefetch_payload(i);
    uint64_t bit = m_has_payloads ? payload_bit(i) : 0;
    alignas(64) uint8_t buffer[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t skip = i - batch;
    uint64_t k = 0;

    while (k < len) {
      const uint64_t take = std::min(len - k, simd_symbols - skip);
      decode_batch<true>(batch, skip, skip + take, bit, buffer);
      std::memcpy(out + k, buffer + skip, take);
      k += take;
      batch += simd_symbols;
      skip = 0;
    }
  }

  uint64_t lce(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_codes.lce(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t limit = std::min(max, m_size - std::max(i, j));
    if (limit == 0) [[unlikely]] return 0;
    const uint64_t wi = class_window(i);
    const uint64_t wj = class_window(j);
    const uint64_t diff = wi ^ wj;
    if ((diff & m_class_mask) != 0) return 0;

    if (m_eager) {
      prefetch_predicted(i, diff == 0 ? predicted_lines : 1);
      prefetch_predicted(j, diff == 0 ? predicted_lines : 1);
    }

    if (diff != 0 || limit <= m_short) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_bit_to_symbol[std::countr_zero(diff)] : m_short);
      if (!window_has_payload(wi, len)) return len;
      return payload_lce(i, j, len);
    }

    if (!m_eager) {
      prefetch_table(i);
      prefetch_table(j);
    }

    const uint64_t vi = class_window(i + m_short);
    const uint64_t vj = class_window(j + m_short);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_short) {
      const uint64_t len = std::min<uint64_t>(limit, m_short + (diff2 != 0 ? m_bit_to_symbol[std::countr_zero(diff2)] : m_short));
      if (!window_has_payload(wi, m_short) && !window_has_payload(vi, len - m_short)) return len;
      return payload_lce(i, j, len);
    }

    return long_lce(i, j, limit);
  }

  uint64_t lce_left(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_codes.lce_left(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min(max, std::min(i, j) + 1);
    if (limit == 0) [[unlikely]] return 0;
    if (std::min(i, j) + 1 < 2 * m_short) [[unlikely]] return slow_lce_left(i, j, limit);
    const uint64_t wi = class_window(i + 1 - m_short);
    const uint64_t wj = class_window(j + 1 - m_short);
    const uint64_t diff = wi ^ wj;
    if ((diff >> ((m_short - 1) * m_width)) != 0) return 0;

    if (m_eager) {
      prefetch_predicted(i, diff == 0 ? -predicted_lines : -1);
      prefetch_predicted(j, diff == 0 ? -predicted_lines : -1);
    }

    if (diff != 0 || limit <= m_short) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_short - 1 - m_bit_to_symbol[63 - std::countl_zero(diff)] : m_short);
      if (!window_has_payload(wi >> ((m_short - len) * m_width), len)) return len;
      return payload_lce_left(i, j, len);
    }

    if (!m_eager) {
      prefetch_table(i);
      prefetch_table(j);
    }

    const uint64_t vi = class_window(i + 1 - 2 * m_short);
    const uint64_t vj = class_window(j + 1 - 2 * m_short);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_short) {
      const uint64_t len = std::min<uint64_t>(limit, m_short + (diff2 != 0 ? m_short - 1 - m_bit_to_symbol[63 - std::countl_zero(diff2)] : m_short));
      const uint64_t rest = len - m_short;
      if (!window_has_payload(wi, m_short) && !window_has_payload(vi >> ((m_short - rest) * m_width), rest)) return len;
      return payload_lce_left(i, j, len);
    }

    return long_lce_left(i, j, limit);
  }

  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  uint64_t hash(uint64_t i, uint64_t len) const {
    const uint64_t h = m_codes.hash(i, len);
    if (!m_has_payloads || len == 0) return h;
    return payload_hash(i, len, h);
  }

 private:
  struct piece_t {
    uint64_t at = 0;
    uint64_t bits = 0;
    std::vector<uint64_t> words;
    std::vector<uint16_t> block_bits;
  };

  struct data_t {
    std::vector<uint8_t> payload;
    std::vector<uint64_t> abs;
    std::vector<uint8_t> rel;
    std::vector<uint8_t> symbol_of;
    std::vector<uint8_t> char_of;
    std::vector<piece_t> pieces;
    std::mutex mutex;
    alignas(64) std::array<uint8_t, 256> len{};
    alignas(64) std::array<uint8_t, 256> mask{};
    alignas(64) std::array<uint8_t, 64> spread{};
    alignas(64) std::array<uint8_t, 64> shifts{};
    alignas(64) std::array<uint8_t, 64> direct_symbol{};
    alignas(64) std::array<uint8_t, 64> direct_char{};
    alignas(64) std::array<uint8_t, 64> bit_to_symbol{};
    std::array<uint8_t, 256> class_of{};
    std::array<uint8_t, 256> payload_of{};
    std::array<uint8_t, 256> class_of_char{};
    std::array<uint8_t, 256> payload_of_char{};
    std::array<uint8_t, 256> len_of_char{};
    std::array<uint8_t, 256> to_symbol{};
    std::array<uint8_t, 256> to_char{};
  };

  void cache_pointers() {
    data_t& d = *m_data;
    m_classes = m_codes.packed_data();
    m_classes_limit = m_classes + (m_size * m_width + 63) / 64 * 8;
    m_payload = d.payload.data();
    m_abs = d.abs.data();
    m_rel = d.rel.data();
    m_len = d.len.data();
    m_mask = d.mask.data();
    m_symbol_of = d.symbol_of.data();
    m_char_of = d.char_of.data();
    m_to_symbol = d.to_symbol.data();
    m_to_char = d.to_char.data();
    m_spread = d.spread.data();
    m_shifts = d.shifts.data();
    m_direct_symbol = d.direct_symbol.data();
    m_direct_char = d.direct_char.data();
    m_bit_to_symbol = d.bit_to_symbol.data();
  }

  static uint64_t table_bits(uint64_t size) {
    return (size / super_symbols + 2) * 64 + (size / block_symbols + 2) * 8 * offset_bytes;
  }

  static uint8_t choose_code(const std::vector<uint64_t>& sorted, uint64_t size, uint64_t penalty,
                             std::vector<uint8_t>& lengths) {
    const uint64_t sigma = sorted.size();
    double best = std::numeric_limits<double>::max();
    uint8_t best_width = 8;
    lengths.clear();

    for (uint8_t width = 1; width <= 8; ++width) {
      const uint64_t classes = uint64_t{1} << width;
      std::vector<uint8_t> candidate;
      double cost = 0;

      if (width > max_split_width) {
        if (classes < sigma) continue;
        candidate.assign(sigma, 0);
        cost = double(width) * double(size);
      } else {
        cost = optimal_lengths(sorted, width, penalty, candidate);
        bool any_payload = false;
        for (uint8_t l : candidate) any_payload = any_payload || l != 0;
        if (any_payload) cost += double(table_bits(size));
      }

      if (cost < best) {
        best = cost;
        best_width = width;
        lengths = candidate;
      }
    }

    if (lengths.empty()) lengths.assign(1, 0);
    return best_width;
  }

  static double optimal_lengths(const std::vector<uint64_t>& sorted, uint8_t width, uint64_t penalty,
                                std::vector<uint8_t>& lengths) {
    const uint64_t m = sorted.size();
    const uint64_t classes = uint64_t{1} << width;
    const double inf = std::numeric_limits<double>::max();
    std::vector<double> prefix(m + 1, 0);
    for (uint64_t p = 0; p < m; ++p) prefix[p + 1] = prefix[p] + double(sorted[p]);
    std::vector<double> f((classes + 1) * (m + 1), inf);
    std::vector<uint8_t> from_len((classes + 1) * (m + 1), 0);
    std::vector<uint16_t> from_p((classes + 1) * (m + 1), 0);
    auto at = [&](uint64_t c, uint64_t p) { return c * (m + 1) + p; };
    f[at(0, 0)] = 0;

    for (uint8_t l = 0; l <= max_payload; ++l) {
      const uint64_t slots = uint64_t{1} << l;
      const double per_symbol = double(width) + double(l) + (l != 0 ? double(penalty) : 0.0);

      for (uint64_t c = 1; c <= classes; ++c) {
        for (uint64_t p = slots; p <= m; ++p) {
          const double base = f[at(c - 1, p - slots)];
          if (base >= inf) continue;
          const double cost = base + (prefix[p] - prefix[p - slots]) * per_symbol;

          if (cost < f[at(c, p)]) {
            f[at(c, p)] = cost;
            from_len[at(c, p)] = l;
            from_p[at(c, p)] = uint16_t(p - slots);
          }
        }

        for (uint64_t q = m > slots ? m - slots + 1 : 0; q < m; ++q) {
          const double base = f[at(c - 1, q)];
          if (base >= inf) continue;
          const double cost = base + (prefix[m] - prefix[q]) * per_symbol;

          if (cost < f[at(c, m)]) {
            f[at(c, m)] = cost;
            from_len[at(c, m)] = l;
            from_p[at(c, m)] = uint16_t(q);
          }
        }
      }
    }

    double best = inf;
    uint64_t best_c = 0;

    for (uint64_t c = 1; c <= classes; ++c) {
      if (f[at(c, m)] < best) {
        best = f[at(c, m)];
        best_c = c;
      }
    }

    lengths.clear();
    if (best >= inf) return inf;
    uint64_t c = best_c;
    uint64_t p = m;

    while (c > 0) {
      lengths.push_back(from_len[at(c, p)]);
      p = from_p[at(c, p)];
      --c;
    }

    std::sort(lengths.begin(), lengths.end());
    double cost = 0;
    uint64_t next = 0;

    for (uint8_t l : lengths) {
      const uint64_t take = std::min<uint64_t>(uint64_t{1} << l, m - next);
      cost += (prefix[next + take] - prefix[next]) * (double(width) + double(l) + (l != 0 ? double(penalty) : 0.0));
      next += take;
    }

    return cost;
  }

  static void write_offset(uint8_t* rel, uint64_t block, uint64_t value) {
    for (uint64_t k = 0; k < offset_bytes; ++k) rel[offset_bytes * block + k] = uint8_t(value >> (8 * k));
  }

  uint64_t block_bit(uint64_t block) const {
    uint32_t value;
    std::memcpy(&value, m_rel + offset_bytes * block, sizeof(value));
    return m_abs[block / blocks_per_super] + (value & ((uint32_t{1} << (8 * offset_bytes)) - 1));
  }

  uint64_t payload_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + prefix_bits(block * block_symbols, i - block * block_symbols);
  }

  uint64_t predicted_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + (((i - block * block_symbols) * m_avg_x256) >> 8);
  }

  void prefetch_table(uint64_t i) const { __builtin_prefetch(m_rel + offset_bytes * (i / block_symbols)); }

  void prefetch_payload(uint64_t i) const { __builtin_prefetch(m_payload + (predicted_bit(i) >> 3)); }

  void prefetch_range(uint64_t bit, uint64_t bits) const {
    const uint8_t* end = m_payload + ((bit + bits) >> 3);
    for (const uint8_t* p = m_payload + (bit >> 3); p <= end; p += 64) __builtin_prefetch(p);
  }

  void prefetch_predicted(uint64_t i, int64_t lines) const {
    const uint64_t byte = predicted_bit(i) >> 3;

    if (lines > 0) {
      for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
    } else {
      for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
    }
  }

  template <bool t_chars>
  uint8_t payload_symbol(uint64_t i, uint8_t c) const {
    prefetch_payload(i);
    const uint64_t bit = payload_bit(i);
    const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
    return (t_chars ? m_char_of : m_symbol_of)[(uint64_t(c) << max_payload) | payload];
  }

  uint64_t class_window(uint64_t i) const {
    const uint64_t bit = i * m_width;
    return (load_u64(m_classes + (bit >> 3)) >> (bit & 7)) & m_short_mask;
  }

  bool window_has_payload(uint64_t window, uint64_t len) const {
    const uint64_t x = window & ((uint64_t{1} << (len * m_width)) - 1);
    const uint64_t t = ((x & m_field_low) | m_field_high) - m_direct_low;
    const uint64_t r = (m_direct_high ? (x & t) : (x | t)) & m_field_high;
    return r != 0;
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t pi = payload_bit(i);
    const uint64_t bits = len <= medium_symbols ? range_bits(i, len) : payload_bit(i + len) - pi;
    if (bits == 0) return len;
    const uint64_t pj = payload_bit(j);
    prefetch_range(pi, bits);
    prefetch_range(pj, bits);
    const uint64_t common = common_prefix(pi, pj, bits);
    if (common == bits) return len;
    return select_forward(i, common);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t first = i + 1 - len;
    const uint64_t ei = payload_bit(i + 1);
    const uint64_t bits = len <= medium_symbols ? range_bits(first, len) : ei - payload_bit(first);
    if (bits == 0) return len;
    const uint64_t ej = payload_bit(j + 1);
    prefetch_range(ei - bits, bits);
    prefetch_range(ej - bits, bits);
    const uint64_t common = common_suffix(ei, ej, bits);
    if (common == bits) return len;
    return select_backward(i, common);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager && m_avg_x256 >= sparse_x256) {
      prefetch_predicted(i, predicted_lines);
      prefetch_predicted(j, predicted_lines);
    }

    const uint64_t skip = 2 * m_short;
    const uint64_t len = skip + m_codes.lce(i + skip, j + skip, limit - skip);
    return payload_lce(i, j, len);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager && m_avg_x256 >= sparse_x256) {
      prefetch_predicted(i, -predicted_lines);
      prefetch_predicted(j, -predicted_lines);
    }

    const uint64_t skip = 2 * m_short;
    const uint64_t len = skip + m_codes.lce_left(i - skip, j - skip, limit - skip);
    return payload_lce_left(i, j, len);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t slow_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    const uint64_t len = m_codes.lce_left(i, j, limit);
    if (len == 0) return 0;
    return payload_lce_left(i, j, len);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_hash(uint64_t i, uint64_t len, uint64_t h) const {
    const uint64_t beg = payload_bit(i);
    const uint64_t bits = payload_bit(i + len) - beg;
    if (bits == 0) return h;
    const uint64_t full = bits / window_bits;
    auto chunk_at = [this](uint64_t bit) { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); };
    uint64_t g = hash_words(full, [&](uint64_t k) { return chunk_at(beg + window_bits * k) & window_mask; });
    const uint64_t rest = bits - window_bits * full;
    if (rest != 0) g = hash_mix(g, chunk_at(beg + window_bits * full) & ((uint64_t{1} << rest) - 1));
    return hash_mix(h, g);
  }

  static uint64_t lane_range(uint64_t skip, uint64_t end) {
    return (end >= simd_symbols ? ~uint64_t{0} : (uint64_t{1} << end) - 1) & (~uint64_t{0} << skip);
  }

  template <bool t_chars>
  void decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint8_t* __restrict out) const {
    const uint8_t* table = t_chars ? m_char_of : m_symbol_of;
    uint64_t bit = position;
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI) || defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
    if (m_width <= max_split_width) {
      alignas(64) uint8_t classes[simd_symbols];
      const uint64_t m = decode_direct(batch, t_chars ? m_direct_char : m_direct_symbol, classes, out) & lane_range(skip, end);

      if (m_has_payloads) {
        uint64_t rest = m;

        while (rest != 0) {
          const uint64_t x = uint64_t(std::countr_zero(rest));
          rest &= rest - 1;
          const uint8_t c = classes[x];
          const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
          bit += m_len[c];
          out[x] = table[(uint64_t(c) << max_payload) | payload];
        }
      }

      position = bit;
      return;
    }
#endif

    packed_text<>::cursor classes = m_codes.cursor_at(batch + skip);

    for (uint64_t x = skip; x < end; ++x) {
      const uint8_t c = classes.next();
      const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
      bit += m_len[c];
      out[x] = table[(uint64_t(c) << max_payload) | payload];
    }

    position = bit;
  }

#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
  __m512i unpack_at(const uint8_t* p) const {
    const __m512i raw = _mm512_loadu_si512(p);
    const __m512i spread = _mm512_permutexvar_epi8(_mm512_load_si512(m_spread), raw);
    const __m512i fields = _mm512_multishift_epi64_epi8(_mm512_load_si512(m_shifts), spread);
    return _mm512_and_si512(fields, _mm512_set1_epi8(char(m_class_mask)));
  }

  __m512i unpack(uint64_t first) const { return unpack_at(m_classes + (first * m_width) / 8); }

  uint64_t decode_direct(uint64_t batch, const uint8_t* direct, uint8_t* classes, uint8_t* __restrict out) const {
    const __m512i cls = unpack(batch);
    _mm512_store_si512(classes, cls);
    _mm512_storeu_si512(out, _mm512_permutexvar_epi8(cls, _mm512_load_si512(direct)));
    if (!m_has_payloads) return 0;
    const __m512i lens = _mm512_permutexvar_epi8(cls, _mm512_load_si512(m_len));
    return _cvtmask64_u64(_mm512_test_epi8_mask(lens, lens));
  }

  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const __m512i zero = _mm512_setzero_si512();
    const uint8_t* p = m_classes + (first * m_width) / 8;
    __m512i acc = zero;
    const __m512i table = _mm512_load_si512(m_len);
    const uint64_t batch_bytes = simd_symbols * m_width / 8;

    for (uint64_t c = 0; c < batches_per_block; ++c) {
      const uint64_t take = std::min<uint64_t>(count > c * simd_symbols ? count - c * simd_symbols : 0, simd_symbols);
      const __mmask64 k = _cvtu64_mask64(take >= simd_symbols ? ~uint64_t{0} : (uint64_t{1} << take) - 1);
      const __m512i cls = unpack_at(std::min(p + c * batch_bytes, m_classes_limit));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, cls, table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i table = _mm512_load_si512(m_len);
    const uint64_t end = i + len;
    __m512i acc = zero;

    for (uint64_t batch = i / simd_symbols * simd_symbols; batch < end; batch += simd_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, simd_symbols);
      const __mmask64 k = _cvtu64_mask64(lane_range(lo, hi));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, unpack(batch), table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    const __mmask64 k = _cvtu64_mask64(lanes);
    const __m512i l = _mm512_maskz_permutexvar_epi8(k, unpack(batch), _mm512_load_si512(m_len));
    _mm512_store_si512(lens, l);
    return uint64_t(_mm512_reduce_add_epi64(_mm512_sad_epu8(l, _mm512_setzero_si512())));
  }
#elif defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
  __m256i unpack_pair(const uint8_t* p) const {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + m_width));
    const __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i pairs = _mm256_shuffle_epi8(raw, _mm256_load_si256(reinterpret_cast<const __m256i*>(m_spread)));
    const __m256i scale = _mm256_load_si256(reinterpret_cast<const __m256i*>(m_shifts));
    return _mm256_srli_epi16(_mm256_mullo_epi16(pairs, scale), 8);
  }

  void unpack_bytes(const uint8_t* p, uint8_t* out) const {
    const __m256i mask = _mm256_set1_epi8(char(m_class_mask));
    const uint64_t step = 2 * m_width;
    const __m256i a = _mm256_packus_epi16(unpack_pair(p), unpack_pair(p + step));
    const __m256i b = _mm256_packus_epi16(unpack_pair(p + 2 * step), unpack_pair(p + 3 * step));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_and_si256(_mm256_permute4x64_epi64(a, 0xd8), mask));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_and_si256(_mm256_permute4x64_epi64(b, 0xd8), mask));
  }

  __m256i lookup(const uint8_t* table, __m256i classes) const {
    const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table)));
    if (m_width <= 4) return _mm256_shuffle_epi8(t0, classes);
    const __m256i lo = _mm256_and_si256(classes, _mm256_set1_epi8(15));
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(classes, 4), _mm256_set1_epi8(15));
    const __m256i t1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 16)));
    const __m256i t2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 32)));
    const __m256i t3 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 48)));
    __m256i r = _mm256_shuffle_epi8(t0, lo);
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t1, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(1)));
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t2, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(2)));
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t3, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(3)));
    return r;
  }

  static __m256i lane_bytes(uint32_t lanes) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(lanes)),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                                           2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
  }

  static uint64_t sum_bytes(__m256i a, __m256i b) {
    const __m256i s = _mm256_add_epi64(_mm256_sad_epu8(a, _mm256_setzero_si256()), _mm256_sad_epu8(b, _mm256_setzero_si256()));
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return uint64_t(_mm_cvtsi128_si64(t)) + uint64_t(_mm_extract_epi64(t, 1));
  }

  uint64_t lens_of(const uint8_t* classes, uint64_t lanes, __m256i& la, __m256i& lb) const {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    la = _mm256_and_si256(lookup(m_len, a), lane_bytes(uint32_t(lanes)));
    lb = _mm256_and_si256(lookup(m_len, b), lane_bytes(uint32_t(lanes >> 32)));
    return sum_bytes(la, lb);
  }

  uint64_t decode_direct(uint64_t batch, const uint8_t* direct, uint8_t* classes, uint8_t* __restrict out) const {
    unpack_bytes(m_classes + (batch * m_width) / 8, classes);
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), lookup(direct, a));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), lookup(direct, b));
    if (!m_has_payloads) return 0;
    const __m256i zero = _mm256_setzero_si256();
    const uint32_t ma = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lookup(m_len, a), zero)));
    const uint32_t mb = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lookup(m_len, b), zero)));
    return ~(uint64_t(ma) | (uint64_t(mb) << 32));
  }

  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const uint64_t batch_bytes = simd_symbols * m_width / 8;
    const uint8_t* p = m_classes + (first * m_width) / 8;
    alignas(64) uint8_t classes[simd_symbols];
    uint64_t bits = 0;

    for (uint64_t c = 0; c < batches_per_block && count > c * simd_symbols; ++c) {
      const uint64_t take = std::min<uint64_t>(count - c * simd_symbols, simd_symbols);
      unpack_bytes(std::min(p + c * batch_bytes, m_classes_limit), classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(0, take), la, lb);
    }

    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const uint64_t end = i + len;
    alignas(64) uint8_t classes[simd_symbols];
    uint64_t bits = 0;

    for (uint64_t batch = i / simd_symbols * simd_symbols; batch < end; batch += simd_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, simd_symbols);
      unpack_bytes(m_classes + (batch * m_width) / 8, classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(lo, hi), la, lb);
    }

    return bits;
  }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    alignas(64) uint8_t classes[simd_symbols];
    unpack_bytes(m_classes + (batch * m_width) / 8, classes);
    __m256i la, lb;
    const uint64_t sum = lens_of(classes, lanes, la, lb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens), la);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens + 32), lb);
    return sum;
  }
#else
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    uint64_t bits = 0;
    for (uint64_t x = 0; x < count; ++x) bits += m_len[m_codes[first + x]];
    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const { return prefix_bits(i, len); }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    uint64_t sum = 0;

    for (uint64_t x = 0; x < simd_symbols; ++x) {
      lens[x] = (lanes >> x) & 1 ? m_len[m_codes[batch + x]] : 0;
      sum += lens[x];
    }

    return sum;
  }
#endif

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_forward(uint64_t i, uint64_t d) const {
    alignas(64) uint8_t lens[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t skip = i - batch;
    uint64_t m = 0;

    for (;;) {
      const uint64_t sum = batch_lens(batch, ~uint64_t{0} << skip, lens);

      if (sum > d) {
        for (uint64_t x = skip;; ++x) {
          if (lens[x] > d) return m;
          d -= lens[x];
          ++m;
        }
      }

      d -= sum;
      m += simd_symbols - skip;
      batch += simd_symbols;
      skip = 0;
    }
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_backward(uint64_t i, uint64_t s) const {
    alignas(64) uint8_t lens[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t last = i - batch;
    uint64_t m = 0;

    for (;;) {
      const uint64_t sum = batch_lens(batch, last == 63 ? ~uint64_t{0} : (uint64_t{1} << (last + 1)) - 1, lens);

      if (sum > s) {
        for (uint64_t x = last;; --x) {
          if (lens[x] > s) return m;
          s -= lens[x];
          ++m;
        }
      }

      s -= sum;
      m += last + 1;
      batch -= simd_symbols;
      last = simd_symbols - 1;
    }
  }

  uint64_t common_prefix(uint64_t a, uint64_t b, uint64_t bits) const {
    uint64_t k = 0;

    while (k + window_bits <= bits) {
      const uint64_t diff = (load_bits(a + k) ^ load_bits(b + k)) & window_mask;
      if (diff != 0) return k + std::countr_zero(diff);
      k += window_bits;
    }

    if (k < bits) {
      const uint64_t diff = (load_bits(a + k) ^ load_bits(b + k)) & ((uint64_t{1} << (bits - k)) - 1);
      if (diff != 0) return k + std::countr_zero(diff);
    }

    return bits;
  }

  uint64_t common_suffix(uint64_t a, uint64_t b, uint64_t bits) const {
    uint64_t k = 0;

    while (k + window_bits <= bits) {
      const uint64_t diff = (load_bits(a - k - window_bits) ^ load_bits(b - k - window_bits)) & window_mask;
      if (diff != 0) return k + std::countl_zero(diff) - (64 - window_bits);
      k += window_bits;
    }

    if (k < bits) {
      const uint64_t rest = bits - k;
      const uint64_t diff = (load_bits(a - bits) ^ load_bits(b - bits)) & ((uint64_t{1} << rest) - 1);
      if (diff != 0) return k + std::countl_zero(diff) - (64 - rest);
    }

    return bits;
  }

  uint64_t load_bits(uint64_t bit) const { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); }

  packed_text<> m_codes;
  std::shared_ptr<data_t> m_data;
  const uint8_t* m_classes = nullptr;
  const uint8_t* m_classes_limit = nullptr;
  const uint8_t* m_payload = nullptr;
  const uint64_t* m_abs = nullptr;
  const uint8_t* m_rel = nullptr;
  const uint8_t* m_len = nullptr;
  const uint8_t* m_mask = nullptr;
  const uint8_t* m_symbol_of = nullptr;
  const uint8_t* m_char_of = nullptr;
  const uint8_t* m_to_symbol = nullptr;
  const uint8_t* m_to_char = nullptr;
  const uint8_t* m_spread = nullptr;
  const uint8_t* m_shifts = nullptr;
  const uint8_t* m_direct_symbol = nullptr;
  const uint8_t* m_direct_char = nullptr;
  const uint8_t* m_bit_to_symbol = nullptr;
  uint64_t m_size = 0;
  uint64_t m_payload_bits = 0;
  uint64_t m_penalty = payload_penalty;
  uint64_t m_short_mask = 0;
  uint64_t m_field_one = 0;
  uint64_t m_field_high = 0;
  uint64_t m_field_low = 0;
  uint64_t m_direct_low = 0;
  uint32_t m_avg_x256 = 0;
  uint16_t m_sigma = 0;
  uint16_t m_direct_classes = 0;
  uint8_t m_width = 0;
  uint8_t m_class_mask = 0;
  uint8_t m_short = 0;
  bool m_direct_high = false;
  bool m_eager = false;
  bool m_has_payloads = false;
};

template <>
class split_text<uint32_t> {
 public:
  using char_type = uint32_t;
  using symbol_type = uint32_t;
  using histogram_t = std::vector<uint64_t>;

  static constexpr bool is_byte_text = false;
  static constexpr uint64_t payload_penalty = 2;
  static constexpr uint64_t simd_symbols = 64;
  static constexpr uint64_t batches_per_block = 4;
  static constexpr uint64_t block_symbols = batches_per_block * simd_symbols;
  static constexpr uint64_t chunk_symbols = block_symbols;
  static constexpr uint64_t super_symbols = uint64_t{1} << 16;
  static constexpr uint64_t blocks_per_super = super_symbols / block_symbols;
  static constexpr uint64_t offset_bytes = 3;
  static constexpr uint64_t max_payload = 32;
  static constexpr uint64_t max_split_width = 6;
  static constexpr uint64_t max_width = 8;
  static constexpr uint64_t padding_bytes = 128;
  static constexpr uint64_t medium_symbols = 192;
  static constexpr uint64_t window_bits = 56;
  static constexpr uint64_t window_mask = (uint64_t{1} << window_bits) - 1;
  static constexpr int64_t predicted_lines = 3;
  static constexpr uint32_t sparse_x256 = 8;
  static constexpr uint64_t eager_denominator = 5;
  static constexpr uint64_t exact_code_symbols = uint64_t{1} << 12;
  static constexpr uint64_t coarse_code_positions = uint64_t{1} << 12;
  static constexpr uint64_t max_classes = uint64_t{1} << max_split_width;
  static constexpr uint64_t max_hist_cells = uint64_t{1} << 24;

  static_assert(super_symbols * max_payload <= uint64_t{1} << (8 * offset_bytes));

  class cursor {
   public:
    cursor() = default;

    cursor(const split_text* text, uint64_t i)
        : m_text(text), m_batch(i / simd_symbols * simd_symbols), m_at(uint32_t(i - m_batch)) {
      m_bit = text->m_has_payloads ? text->payload_bit(i) : 0;
      text->decode_batch(m_batch, m_at, simd_symbols, m_bit, m_buffer);
    }

    uint32_t next() {
      if (m_at == simd_symbols) [[unlikely]] {
        m_batch += simd_symbols;
        m_at = 0;
        m_text->decode_batch(m_batch, 0, simd_symbols, m_bit, m_buffer);
      }

      return m_buffer[m_at++];
    }

   private:
    alignas(64) uint32_t m_buffer[simd_symbols];
    const split_text* m_text = nullptr;
    uint64_t m_batch = 0;
    uint64_t m_bit = 0;
    uint32_t m_at = 0;
  };

  split_text() = default;

  split_text(const uint32_t* data, uint64_t size, uint64_t sigma, int threads = omp_get_max_threads())
      : split_text(count_symbols(data, size, sigma, threads), size) {
    pack(data, size, 0, threads);
    finish(threads);
  }

  split_text(const histogram_t& histogram, uint64_t size, uint64_t penalty = payload_penalty)
      : m_size(size), m_penalty(penalty) {
    m_data = std::make_shared<data_t>();
    data_t& d = *m_data;
    const uint64_t sigma = histogram.size();
    m_sigma = std::max<uint64_t>(sigma, 1);
    resize_no_init(d.value_of, sigma);
    std::iota(d.value_of.begin(), d.value_of.end(), uint32_t{0});
    std::stable_sort(d.value_of.begin(), d.value_of.end(),
                     [&](uint32_t a, uint32_t b) { return histogram[a] > histogram[b]; });
    std::vector<uint64_t> sorted;
    resize_no_init(sorted, sigma);
    for (uint64_t r = 0; r < sigma; ++r) sorted[r] = histogram[d.value_of[r]];
    std::vector<uint8_t> lengths;
    m_width = choose_code(sorted, size, penalty, lengths);
    const uint64_t classes = uint64_t{1} << m_width;
    resize_no_init(d.code_of, sigma);
    uint64_t next = 0;
    uint64_t payload_chars = 0;
    m_direct_classes = 0;

    for (uint64_t c = 0; c < lengths.size(); ++c) {
      const uint64_t take = std::min<uint64_t>(uint64_t{1} << lengths[c], sigma - std::min(sigma, next));
      d.len[c] = lengths[c];
      d.mask[c] = (uint64_t{1} << lengths[c]) - 1;
      d.start[c] = uint32_t(next);
      m_has_payloads = m_has_payloads || lengths[c] != 0;
      if (lengths[c] == 0) m_direct_classes = uint16_t(c + 1);

      for (uint64_t p = 0; p < take; ++p) {
        d.code_of[d.value_of[next + p]] = (p << 8) | c;
        if (lengths[c] != 0) payload_chars += sorted[next + p];
      }

      next += take;
    }

    m_eager = size != 0 && payload_chars * eager_denominator >= size;

    for (uint64_t k = 0; k < 64; ++k) {
#if defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
      const uint64_t bit = ((k >> 1) & 7) * m_width;
      const uint64_t scale = uint64_t{256} >> (bit & 7);
      d.spread[k] = uint8_t((bit >> 3) + (k & 1));
      d.shifts[k] = uint8_t((k & 1) != 0 ? scale >> 8 : scale);
#else
      d.spread[k] = uint8_t((k >> 3) * m_width + (k & 7));
      d.shifts[k] = uint8_t((k & 7) * m_width);
#endif
      d.bit_to_symbol[k] = uint8_t(k / m_width);
    }

    m_class_mask = uint8_t(classes - 1);
    m_short = uint8_t(std::min<uint64_t>(window_bits / m_width, simd_symbols - 1));
    m_short_mask = (uint64_t{1} << (m_short * m_width)) - 1;

    for (uint64_t field = 0; field * m_width + m_width <= 64; ++field) {
      m_field_one |= uint64_t{1} << (field * m_width);
      m_field_high |= uint64_t{1} << (field * m_width + m_width - 1);
    }

    m_field_low = m_field_high - m_field_one;
    const uint64_t direct = std::min<uint64_t>(m_direct_classes, classes - 1);
    m_direct_high = ((direct >> (m_width - 1)) & 1) != 0;
    m_direct_low = m_field_one * (direct & ((uint64_t{1} << (m_width - 1)) - 1));
    packed_text<>::histogram_t class_histogram{};
    for (uint64_t c = 0; c < classes; ++c) class_histogram[c] = 1;
    m_codes = packed_text<>(class_histogram, size);

    if (m_has_payloads) {
      d.abs.assign(size / super_symbols + 2, 0);
      d.rel.assign(offset_bytes * (size / block_symbols + 2) + 8, 0);
    }

    d.payload.assign(padding_bytes, 0);
    cache_pointers();
  }

  static histogram_t count_symbols(const uint32_t* data, uint64_t size, uint64_t sigma,
                                   int threads = omp_get_max_threads()) {
    histogram_t histogram(sigma, 0);

    if (sigma * uint64_t(std::max(threads, 1)) <= max_hist_cells) {
      std::vector<histogram_t> partial(std::max(threads, 1), histogram_t(sigma, 0));

#pragma omp parallel num_threads(threads)
      {
        histogram_t& local = partial[omp_get_thread_num()];
#pragma omp for
        for (uint64_t i = 0; i < size; ++i) ++local[data[i]];
      }

      for (const histogram_t& local : partial) {
        for (uint64_t c = 0; c < sigma; ++c) histogram[c] += local[c];
      }
    } else {
#pragma omp parallel for num_threads(threads)
      for (uint64_t i = 0; i < size; ++i) {
        std::atomic_ref<uint64_t>(histogram[data[i]]).fetch_add(1, std::memory_order_relaxed);
      }
    }

    return histogram;
  }

  void pack(const uint32_t* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) {
    const uint64_t* code_of = m_data->code_of.data();
    m_codes.pack_symbols(length, at, [code_of, data](uint64_t i) { return code_of[data[i]] & 0xFF; }, threads);
    if (!m_has_payloads || length == 0) return;
    const uint8_t* len_of_class = m_data->len.data();
    const uint64_t blocks = (length + block_symbols - 1) / block_symbols;
    const uint64_t parts = std::min<uint64_t>(blocks, std::max(threads, 1));

#pragma omp parallel for num_threads(threads)
    for (uint64_t part = 0; part < parts; ++part) {
      const uint64_t first_block = blocks * part / parts;
      const uint64_t last_block = blocks * (part + 1) / parts;
      const uint64_t beg = first_block * block_symbols;
      const uint64_t end = std::min(length, last_block * block_symbols);
      piece_t piece;
      piece.at = at + beg;
      piece.block_bits.assign(last_block - first_block, 0);
      uint64_t total = 0;

      for (uint64_t b = first_block; b < last_block; ++b) {
        const uint64_t block_end = std::min(end, (b + 1) * block_symbols);
        uint64_t bits = 0;
        for (uint64_t i = b * block_symbols; i < block_end; ++i) bits += len_of_class[code_of[data[i]] & 0xFF];
        piece.block_bits[b - first_block] = uint16_t(bits);
        total += bits;
      }

      piece.bits = total;
      resize_no_init(piece.words, (total + 63) / 64 + 1);
      uint64_t word = 0;
      uint64_t acc = 0;
      uint64_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint64_t code = code_of[data[i]];
        const uint64_t value = code >> 8;
        const uint64_t len = len_of_class[code & 0xFF];
        acc |= value << acc_bits;
        acc_bits += len;

        if (acc_bits >= 64) {
          piece.words[word++] = acc;
          acc_bits -= 64;
          acc = acc_bits == 0 ? 0 : value >> (len - acc_bits);
        }
      }

      if (acc_bits != 0) piece.words[word++] = acc;
      piece.words.resize(word);
      std::lock_guard<std::mutex> lock(m_data->mutex);
      m_data->pieces.push_back(std::move(piece));
    }
  }

  void finish(int threads = omp_get_max_threads()) {
    data_t& d = *m_data;

    d.code_of = std::vector<uint64_t>();

    if (!m_has_payloads) {
      d.pieces = decltype(d.pieces)();
      cache_pointers();
      return;
    }

    std::sort(d.pieces.begin(), d.pieces.end(), [](const piece_t& a, const piece_t& b) { return a.at < b.at; });
    const uint64_t pieces = d.pieces.size();
    std::vector<uint64_t> offset(pieces + 1, 0);
    for (uint64_t p = 0; p < pieces; ++p) offset[p + 1] = offset[p] + d.pieces[p].bits;
    const uint64_t total = offset[pieces];
    m_payload_bits = total;
    m_avg_x256 = uint32_t(m_size == 0 ? 0 : (total * 256 + m_size / 2) / m_size);
    resize_no_init(d.payload, (total + 63) / 64 * 8 + padding_bytes);
    advise_huge_pages(d.payload.data(), d.payload.size());
    parallel_memset(d.payload.data(), 0, d.payload.size(), threads);

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        if (block % blocks_per_super == 0) d.abs[block / blocks_per_super] = cum;
        cum += piece.block_bits[k];
      }
    }

    const uint64_t end_block = (m_size + block_symbols - 1) / block_symbols;
    if (end_block % blocks_per_super == 0) d.abs[end_block / blocks_per_super] = total;

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        write_offset(d.rel.data(), block, cum - d.abs[block / blocks_per_super]);
        cum += piece.block_bits[k];
      }
    }

    write_offset(d.rel.data(), end_block, total - d.abs[end_block / blocks_per_super]);
    uint64_t* out = reinterpret_cast<uint64_t*>(d.payload.data());

#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      if (piece.bits == 0) continue;
      const uint64_t first = offset[p] >> 6;
      const uint64_t last = (offset[p] + piece.bits - 1) >> 6;
      const uint64_t shift = offset[p] & 63;
      const uint64_t words = (piece.bits + 63) / 64;
      const uint64_t tail = piece.bits & 63;
      uint64_t carry = 0;

      for (uint64_t w = first; w <= last; ++w) {
        const uint64_t k = w - first;
        uint64_t x = k < words ? piece.words[k] : 0;
        if (k + 1 == words && tail != 0) x &= (uint64_t{1} << tail) - 1;
        const uint64_t value = (x << shift) | carry;
        carry = shift == 0 ? 0 : x >> (64 - shift);
        if (w == first || w == last) std::atomic_ref<uint64_t>(out[w]).fetch_or(value, std::memory_order_relaxed);
        else out[w] = value;
      }
    }

    d.pieces = decltype(d.pieces)();
    cache_pointers();
  }

  uint64_t size() const { return m_size; }
  uint64_t sigma() const { return m_sigma; }
  uint8_t width() const { return m_width; }
  uint64_t penalty() const { return m_penalty; }
  bool has_payloads() const { return m_has_payloads; }
  uint64_t payload_bits() const { return m_payload_bits; }
  const uint8_t* class_lengths() const { return m_len; }

  uint64_t size_in_bytes() const {
    uint64_t bytes = sizeof(*this) + m_codes.size_in_bytes();
    if (m_data) {
      bytes += sizeof(data_t) + m_data->payload.size() + m_data->abs.size() * 8 + m_data->rel.size() +
               m_data->value_of.size() * sizeof(uint32_t) + m_data->code_of.size() * sizeof(uint64_t);
    }
    return bytes;
  }

  static uint64_t size_in_bytes_for(const histogram_t& histogram, uint64_t size, uint64_t penalty = payload_penalty) {
    std::vector<uint64_t> sorted(histogram.begin(), histogram.end());
    std::sort(sorted.begin(), sorted.end(), std::greater<uint64_t>());
    std::vector<uint8_t> lengths;
    const uint8_t width = choose_code(sorted, size, penalty, lengths);
    uint64_t payload_bits = 0;
    uint64_t next = 0;
    bool any_payload = false;

    for (uint8_t l : lengths) {
      any_payload = any_payload || l != 0;
      for (uint64_t p = 0; p < (uint64_t{1} << l) && next < sorted.size(); ++p, ++next) payload_bits += sorted[next] * l;
    }

    uint64_t bytes = sizeof(split_text) + sizeof(packed_text<>) + (size * width + 63) / 64 * 8 + packed_text<>::padding_bytes +
                     sizeof(data_t) + histogram.size() * sizeof(uint32_t);

    if (!any_payload) return bytes + padding_bytes;
    return bytes + (payload_bits + 63) / 64 * 8 + padding_bytes + (size / super_symbols + 2) * 8 +
           offset_bytes * (size / block_symbols + 2) + 8;
  }

  uint32_t operator[](uint64_t i) const {
    const uint8_t c = m_codes[i];
    if (m_len[c] == 0) return m_start[c];
    return payload_symbol(i, c);
  }

  char_type char_at(uint64_t i) const { return m_value_of[(*this)[i]]; }
  bool less_char(uint64_t i, uint64_t j) const { return (*this)[i] < (*this)[j]; }
  uint32_t to_char(uint32_t symbol) const { return m_value_of[symbol]; }
  cursor cursor_at(uint64_t i) const { return cursor(this, i); }

  void extract(uint64_t i, uint64_t len, uint32_t* __restrict out) const {
    if (len == 0) return;
    if (m_has_payloads) prefetch_payload(i);
    uint64_t bit = m_has_payloads ? payload_bit(i) : 0;
    alignas(64) uint32_t buffer[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t skip = i - batch;
    uint64_t k = 0;

    while (k < len) {
      const uint64_t take = std::min(len - k, simd_symbols - skip);
      decode_batch(batch, skip, skip + take, bit, buffer);
      std::memcpy(out + k, buffer + skip, take * sizeof(uint32_t));
      k += take;
      batch += simd_symbols;
      skip = 0;
    }
  }

  uint64_t lce(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_codes.lce(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t limit = std::min(max, m_size - std::max(i, j));
    if (limit == 0) [[unlikely]] return 0;
    const uint64_t wi = class_window(i);
    const uint64_t wj = class_window(j);
    const uint64_t diff = wi ^ wj;
    if ((diff & m_class_mask) != 0) return 0;

    if (m_eager) {
      prefetch_predicted(i, diff == 0 ? predicted_lines : 1);
      prefetch_predicted(j, diff == 0 ? predicted_lines : 1);
    }

    if (diff != 0 || limit <= m_short) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_bit_to_symbol[std::countr_zero(diff)] : m_short);
      if (!window_has_payload(wi, len)) return len;
      return payload_lce(i, j, len);
    }

    if (!m_eager) {
      prefetch_table(i);
      prefetch_table(j);
    }

    const uint64_t vi = class_window(i + m_short);
    const uint64_t vj = class_window(j + m_short);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_short) {
      const uint64_t len = std::min<uint64_t>(limit, m_short + (diff2 != 0 ? m_bit_to_symbol[std::countr_zero(diff2)] : m_short));
      if (!window_has_payload(wi, m_short) && !window_has_payload(vi, len - m_short)) return len;
      return payload_lce(i, j, len);
    }

    return long_lce(i, j, limit);
  }

  uint64_t lce_left(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_codes.lce_left(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min(max, std::min(i, j) + 1);
    if (limit == 0) [[unlikely]] return 0;
    if (std::min(i, j) + 1 < 2 * m_short) [[unlikely]] return slow_lce_left(i, j, limit);
    const uint64_t wi = class_window(i + 1 - m_short);
    const uint64_t wj = class_window(j + 1 - m_short);
    const uint64_t diff = wi ^ wj;
    if ((diff >> ((m_short - 1) * m_width)) != 0) return 0;

    if (m_eager) {
      prefetch_predicted(i, diff == 0 ? -predicted_lines : -1);
      prefetch_predicted(j, diff == 0 ? -predicted_lines : -1);
    }

    if (diff != 0 || limit <= m_short) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_short - 1 - m_bit_to_symbol[63 - std::countl_zero(diff)] : m_short);
      if (!window_has_payload(wi >> ((m_short - len) * m_width), len)) return len;
      return payload_lce_left(i, j, len);
    }

    if (!m_eager) {
      prefetch_table(i);
      prefetch_table(j);
    }

    const uint64_t vi = class_window(i + 1 - 2 * m_short);
    const uint64_t vj = class_window(j + 1 - 2 * m_short);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_short) {
      const uint64_t len = std::min<uint64_t>(limit, m_short + (diff2 != 0 ? m_short - 1 - m_bit_to_symbol[63 - std::countl_zero(diff2)] : m_short));
      const uint64_t rest = len - m_short;
      if (!window_has_payload(wi, m_short) && !window_has_payload(vi >> ((m_short - rest) * m_width), rest)) return len;
      return payload_lce_left(i, j, len);
    }

    return long_lce_left(i, j, limit);
  }

  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  uint64_t hash(uint64_t i, uint64_t len) const {
    const uint64_t h = m_codes.hash(i, len);
    if (!m_has_payloads || len == 0) return h;
    return payload_hash(i, len, h);
  }

 private:
  struct piece_t {
    uint64_t at = 0;
    uint64_t bits = 0;
    std::vector<uint64_t> words;
    std::vector<uint16_t> block_bits;
  };

  struct data_t {
    std::vector<uint8_t> payload;
    std::vector<uint64_t> abs;
    std::vector<uint8_t> rel;
    std::vector<uint32_t> value_of;
    std::vector<uint64_t> code_of;
    std::vector<piece_t> pieces;
    std::mutex mutex;
    alignas(64) std::array<uint8_t, 256> len{};
    alignas(64) std::array<uint64_t, 256> mask{};
    alignas(64) std::array<uint32_t, 256> start{};
    alignas(64) std::array<uint8_t, 64> spread{};
    alignas(64) std::array<uint8_t, 64> shifts{};
    alignas(64) std::array<uint8_t, 64> bit_to_symbol{};
  };

  struct code_grid {
    std::vector<uint64_t> pos;
    std::vector<uint64_t> prefix;
    uint64_t exact = 0;
    uint64_t step = 1;
    uint8_t step_shift = 0;

    explicit code_grid(const std::vector<uint64_t>& sorted) {
      const uint64_t sigma = sorted.size();
      exact = std::min<uint64_t>(sigma, exact_code_symbols);

      while ((sigma - exact) / step > coarse_code_positions) {
        step <<= 1;
        ++step_shift;
      }

      for (uint64_t p = 0; p <= exact; ++p) pos.push_back(p);
      for (uint64_t p = exact + step; p < sigma; p += step) pos.push_back(p);
      if (pos.back() != sigma) pos.push_back(sigma);
      prefix.assign(pos.size(), 0);
      uint64_t sum = 0;
      uint64_t k = 0;

      for (uint64_t r = 0; r <= sigma; ++r) {
        while (k < pos.size() && pos[k] == r) prefix[k++] = sum;
        if (r < sigma) sum += sorted[r];
      }
    }

    int64_t index_of(uint64_t p) const {
      if (p <= exact) return int64_t(p);
      if (p == pos.back()) return int64_t(pos.size()) - 1;
      if (((p - exact) & (step - 1)) != 0) return -1;
      const uint64_t k = exact + ((p - exact) >> step_shift);
      return k < pos.size() && pos[k] == p ? int64_t(k) : -1;
    }
  };

  void cache_pointers() {
    data_t& d = *m_data;
    m_classes = m_codes.packed_data();
    m_classes_limit = m_classes + (m_size * m_width + 63) / 64 * 8;
    m_payload = d.payload.data();
    m_abs = d.abs.data();
    m_rel = d.rel.data();
    m_len = d.len.data();
    m_mask = d.mask.data();
    m_start = d.start.data();
    m_value_of = d.value_of.data();
    m_spread = d.spread.data();
    m_shifts = d.shifts.data();
    m_bit_to_symbol = d.bit_to_symbol.data();
  }

  static uint64_t table_bits(uint64_t size) {
    return (size / super_symbols + 2) * 64 + (size / block_symbols + 2) * 8 * offset_bytes;
  }

  static uint8_t choose_code(const std::vector<uint64_t>& sorted, uint64_t size, uint64_t penalty,
                             std::vector<uint8_t>& lengths) {
    const uint64_t sigma = sorted.size();
    lengths.clear();

    if (sigma <= 1) {
      lengths.assign(1, 0);
      return 1;
    }

    const code_grid grid(sorted);
    const uint64_t num_pos = grid.pos.size();
    const uint64_t last = num_pos - 1;
    const double total = double(grid.prefix[last]);
    const double inf = std::numeric_limits<double>::max();
    std::vector<double> f((max_classes + 1) * num_pos, inf);
    std::vector<uint8_t> from_len((max_classes + 1) * num_pos, 0);
    std::vector<uint32_t> from_pos((max_classes + 1) * num_pos, 0);
    auto at = [&](uint64_t c, uint64_t k) { return c * num_pos + k; };
    f[at(0, 0)] = 0;

    for (uint64_t c = 0; c < max_classes; ++c) {
      for (uint64_t k = 0; k < last; ++k) {
        const double base = f[at(c, k)];
        if (base >= inf) continue;
        const uint64_t p = grid.pos[k];

        for (uint8_t l = 0; l <= max_payload; ++l) {
          const uint64_t slots = uint64_t{1} << l;
          const double per_symbol = double(l) + (l != 0 ? double(penalty) : 0.0);

          if (p + slots >= sigma) {
            const double cost = base + double(grid.prefix[last] - grid.prefix[k]) * per_symbol;

            if (cost < f[at(c + 1, last)]) {
              f[at(c + 1, last)] = cost;
              from_len[at(c + 1, last)] = l;
              from_pos[at(c + 1, last)] = uint32_t(k);
            }

            break;
          }

          const int64_t t = grid.index_of(p + slots);
          if (t < 0) continue;
          const double cost = base + double(grid.prefix[t] - grid.prefix[k]) * per_symbol;

          if (cost < f[at(c + 1, uint64_t(t))]) {
            f[at(c + 1, uint64_t(t))] = cost;
            from_len[at(c + 1, uint64_t(t))] = l;
            from_pos[at(c + 1, uint64_t(t))] = uint32_t(k);
          }
        }
      }
    }

    double best = inf;
    uint8_t best_width = max_width;
    uint64_t best_classes = 0;

    for (uint8_t width = 1; width <= max_width; ++width) {
      const uint64_t classes = uint64_t{1} << width;

      if (width > max_split_width) {
        if (classes < sigma) continue;
        const double cost = double(width) * double(size);

        if (cost < best) {
          best = cost;
          best_width = width;
          best_classes = 0;
        }

        continue;
      }

      uint64_t c_min = 0;

      for (uint64_t c = 1; c <= classes; ++c) {
        if (f[at(c, last)] < inf && (c_min == 0 || f[at(c, last)] < f[at(c_min, last)])) c_min = c;
      }

      if (c_min == 0) continue;
      const double cost = f[at(c_min, last)] + double(width) * total +
                          (f[at(c_min, last)] > 0 ? double(table_bits(size)) : 0.0);

      if (cost < best) {
        best = cost;
        best_width = width;
        best_classes = c_min;
      }
    }

    if (best_classes == 0) {
      lengths.assign(sigma, 0);
      return best_width;
    }

    uint64_t c = best_classes;
    uint64_t k = last;

    while (c > 0) {
      lengths.push_back(from_len[at(c, k)]);
      k = from_pos[at(c, k)];
      --c;
    }

    std::sort(lengths.begin(), lengths.end());
    return best_width;
  }

  static void write_offset(uint8_t* rel, uint64_t block, uint64_t value) {
    for (uint64_t k = 0; k < offset_bytes; ++k) rel[offset_bytes * block + k] = uint8_t(value >> (8 * k));
  }

  uint64_t block_bit(uint64_t block) const {
    uint32_t value;
    std::memcpy(&value, m_rel + offset_bytes * block, sizeof(value));
    return m_abs[block / blocks_per_super] + (value & ((uint32_t{1} << (8 * offset_bytes)) - 1));
  }

  uint64_t payload_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + prefix_bits(block * block_symbols, i - block * block_symbols);
  }

  uint64_t predicted_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + (((i - block * block_symbols) * m_avg_x256) >> 8);
  }

  void prefetch_table(uint64_t i) const { __builtin_prefetch(m_rel + offset_bytes * (i / block_symbols)); }

  void prefetch_payload(uint64_t i) const { __builtin_prefetch(m_payload + (predicted_bit(i) >> 3)); }

  void prefetch_range(uint64_t bit, uint64_t bits) const {
    const uint8_t* end = m_payload + ((bit + bits) >> 3);
    for (const uint8_t* p = m_payload + (bit >> 3); p <= end; p += 64) __builtin_prefetch(p);
  }

  void prefetch_predicted(uint64_t i, int64_t lines) const {
    const uint64_t byte = predicted_bit(i) >> 3;

    if (lines > 0) {
      for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
    } else {
      for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
    }
  }

  uint32_t payload_symbol(uint64_t i, uint8_t c) const {
    prefetch_payload(i);
    const uint64_t bit = payload_bit(i);
    const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
    return uint32_t(m_start[c] + payload);
  }

  uint64_t class_window(uint64_t i) const {
    const uint64_t bit = i * m_width;
    return (load_u64(m_classes + (bit >> 3)) >> (bit & 7)) & m_short_mask;
  }

  bool window_has_payload(uint64_t window, uint64_t len) const {
    const uint64_t x = window & ((uint64_t{1} << (len * m_width)) - 1);
    const uint64_t t = ((x & m_field_low) | m_field_high) - m_direct_low;
    const uint64_t r = (m_direct_high ? (x & t) : (x | t)) & m_field_high;
    return r != 0;
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t pi = payload_bit(i);
    const uint64_t bits = len <= medium_symbols ? range_bits(i, len) : payload_bit(i + len) - pi;
    if (bits == 0) return len;
    const uint64_t pj = payload_bit(j);
    prefetch_range(pi, bits);
    prefetch_range(pj, bits);
    const uint64_t common = common_prefix(pi, pj, bits);
    if (common == bits) return len;
    return select_forward(i, common);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t first = i + 1 - len;
    const uint64_t ei = payload_bit(i + 1);
    const uint64_t bits = len <= medium_symbols ? range_bits(first, len) : ei - payload_bit(first);
    if (bits == 0) return len;
    const uint64_t ej = payload_bit(j + 1);
    prefetch_range(ei - bits, bits);
    prefetch_range(ej - bits, bits);
    const uint64_t common = common_suffix(ei, ej, bits);
    if (common == bits) return len;
    return select_backward(i, common);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager && m_avg_x256 >= sparse_x256) {
      prefetch_predicted(i, predicted_lines);
      prefetch_predicted(j, predicted_lines);
    }

    uint64_t pi = payload_bit(i);
    uint64_t pj = payload_bit(j);
    uint64_t k = 0;

    while (k < limit) {
      const uint64_t step = std::min(limit - k, block_symbols);
      const uint64_t len = m_codes.lce(i + k, j + k, step);
      const uint64_t bits = range_bits(i + k, len);
      const uint64_t common = common_prefix(pi, pj, bits);
      if (common < bits) return k + select_forward(i + k, common);
      if (len < step) return k + len;
      pi += bits;
      pj += bits;
      k += step;
    }

    return limit;
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager && m_avg_x256 >= sparse_x256) {
      prefetch_predicted(i, -predicted_lines);
      prefetch_predicted(j, -predicted_lines);
    }

    uint64_t ei = payload_bit(i + 1);
    uint64_t ej = payload_bit(j + 1);
    uint64_t k = 0;

    while (k < limit) {
      const uint64_t step = std::min(limit - k, block_symbols);
      const uint64_t len = m_codes.lce_left(i - k, j - k, step);
      const uint64_t bits = len == 0 ? 0 : range_bits(i - k + 1 - len, len);
      const uint64_t common = common_suffix(ei, ej, bits);
      if (common < bits) return k + select_backward(i - k, common);
      if (len < step) return k + len;
      ei -= bits;
      ej -= bits;
      k += step;
    }

    return limit;
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t slow_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    return long_lce_left(i, j, limit);
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_hash(uint64_t i, uint64_t len, uint64_t h) const {
    const uint64_t beg = payload_bit(i);
    const uint64_t bits = payload_bit(i + len) - beg;
    if (bits == 0) return h;
    const uint64_t full = bits / window_bits;
    auto chunk_at = [this](uint64_t bit) { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); };
    uint64_t g = hash_words(full, [&](uint64_t k) { return chunk_at(beg + window_bits * k) & window_mask; });
    const uint64_t rest = bits - window_bits * full;
    if (rest != 0) g = hash_mix(g, chunk_at(beg + window_bits * full) & ((uint64_t{1} << rest) - 1));
    return hash_mix(h, g);
  }

  static uint64_t lane_range(uint64_t skip, uint64_t end) {
    return (end >= simd_symbols ? ~uint64_t{0} : (uint64_t{1} << end) - 1) & (~uint64_t{0} << skip);
  }

  void decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint32_t* __restrict out) const {
    uint64_t bit = position;
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI) || defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
    if (m_width <= max_split_width) {
      alignas(64) uint8_t classes[simd_symbols];
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
      _mm512_store_si512(classes, unpack(batch));
#else
      unpack_bytes(m_classes + (batch * m_width) / 8, classes);
#endif

      for (uint64_t x = skip; x < end; ++x) {
        const uint8_t c = classes[x];
        const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
        bit += m_len[c];
        out[x] = uint32_t(m_start[c] + payload);
      }

      position = bit;
      return;
    }
#endif

    packed_text<>::cursor classes = m_codes.cursor_at(batch + skip);

    for (uint64_t x = skip; x < end; ++x) {
      const uint8_t c = classes.next();
      const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_mask[c];
      bit += m_len[c];
      out[x] = uint32_t(m_start[c] + payload);
    }

    position = bit;
  }

#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
  __m512i unpack_at(const uint8_t* p) const {
    const __m512i raw = _mm512_loadu_si512(p);
    const __m512i spread = _mm512_permutexvar_epi8(_mm512_load_si512(m_spread), raw);
    const __m512i fields = _mm512_multishift_epi64_epi8(_mm512_load_si512(m_shifts), spread);
    return _mm512_and_si512(fields, _mm512_set1_epi8(char(m_class_mask)));
  }

  __m512i unpack(uint64_t first) const { return unpack_at(m_classes + (first * m_width) / 8); }

  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const __m512i zero = _mm512_setzero_si512();
    const uint8_t* p = m_classes + (first * m_width) / 8;
    __m512i acc = zero;
    const __m512i table = _mm512_load_si512(m_len);
    const uint64_t batch_bytes = simd_symbols * m_width / 8;

    for (uint64_t c = 0; c < batches_per_block; ++c) {
      const uint64_t take = std::min<uint64_t>(count > c * simd_symbols ? count - c * simd_symbols : 0, simd_symbols);
      const __mmask64 k = _cvtu64_mask64(take >= simd_symbols ? ~uint64_t{0} : (uint64_t{1} << take) - 1);
      const __m512i cls = unpack_at(std::min(p + c * batch_bytes, m_classes_limit));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, cls, table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i table = _mm512_load_si512(m_len);
    const uint64_t end = i + len;
    __m512i acc = zero;

    for (uint64_t batch = i / simd_symbols * simd_symbols; batch < end; batch += simd_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, simd_symbols);
      const __mmask64 k = _cvtu64_mask64(lane_range(lo, hi));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, unpack(batch), table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    const __mmask64 k = _cvtu64_mask64(lanes);
    const __m512i l = _mm512_maskz_permutexvar_epi8(k, unpack(batch), _mm512_load_si512(m_len));
    _mm512_store_si512(lens, l);
    return uint64_t(_mm512_reduce_add_epi64(_mm512_sad_epu8(l, _mm512_setzero_si512())));
  }
#elif defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
  __m256i unpack_pair(const uint8_t* p) const {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + m_width));
    const __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i pairs = _mm256_shuffle_epi8(raw, _mm256_load_si256(reinterpret_cast<const __m256i*>(m_spread)));
    const __m256i scale = _mm256_load_si256(reinterpret_cast<const __m256i*>(m_shifts));
    return _mm256_srli_epi16(_mm256_mullo_epi16(pairs, scale), 8);
  }

  void unpack_bytes(const uint8_t* p, uint8_t* out) const {
    const __m256i mask = _mm256_set1_epi8(char(m_class_mask));
    const uint64_t step = 2 * m_width;
    const __m256i a = _mm256_packus_epi16(unpack_pair(p), unpack_pair(p + step));
    const __m256i b = _mm256_packus_epi16(unpack_pair(p + 2 * step), unpack_pair(p + 3 * step));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_and_si256(_mm256_permute4x64_epi64(a, 0xd8), mask));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_and_si256(_mm256_permute4x64_epi64(b, 0xd8), mask));
  }

  __m256i lookup(const uint8_t* table, __m256i classes) const {
    const __m256i t0 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table)));
    if (m_width <= 4) return _mm256_shuffle_epi8(t0, classes);
    const __m256i lo = _mm256_and_si256(classes, _mm256_set1_epi8(15));
    const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(classes, 4), _mm256_set1_epi8(15));
    const __m256i t1 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 16)));
    const __m256i t2 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 32)));
    const __m256i t3 = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(table + 48)));
    __m256i r = _mm256_shuffle_epi8(t0, lo);
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t1, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(1)));
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t2, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(2)));
    r = _mm256_blendv_epi8(r, _mm256_shuffle_epi8(t3, lo), _mm256_cmpeq_epi8(hi, _mm256_set1_epi8(3)));
    return r;
  }

  static __m256i lane_bytes(uint32_t lanes) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(lanes)),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                                           2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
  }

  static uint64_t sum_bytes(__m256i a, __m256i b) {
    const __m256i s = _mm256_add_epi64(_mm256_sad_epu8(a, _mm256_setzero_si256()), _mm256_sad_epu8(b, _mm256_setzero_si256()));
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return uint64_t(_mm_cvtsi128_si64(t)) + uint64_t(_mm_extract_epi64(t, 1));
  }

  uint64_t lens_of(const uint8_t* classes, uint64_t lanes, __m256i& la, __m256i& lb) const {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    la = _mm256_and_si256(lookup(m_len, a), lane_bytes(uint32_t(lanes)));
    lb = _mm256_and_si256(lookup(m_len, b), lane_bytes(uint32_t(lanes >> 32)));
    return sum_bytes(la, lb);
  }

  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const uint64_t batch_bytes = simd_symbols * m_width / 8;
    const uint8_t* p = m_classes + (first * m_width) / 8;
    alignas(64) uint8_t classes[simd_symbols];
    uint64_t bits = 0;

    for (uint64_t c = 0; c < batches_per_block && count > c * simd_symbols; ++c) {
      const uint64_t take = std::min<uint64_t>(count - c * simd_symbols, simd_symbols);
      unpack_bytes(std::min(p + c * batch_bytes, m_classes_limit), classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(0, take), la, lb);
    }

    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const uint64_t end = i + len;
    alignas(64) uint8_t classes[simd_symbols];
    uint64_t bits = 0;

    for (uint64_t batch = i / simd_symbols * simd_symbols; batch < end; batch += simd_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, simd_symbols);
      unpack_bytes(m_classes + (batch * m_width) / 8, classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(lo, hi), la, lb);
    }

    return bits;
  }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    alignas(64) uint8_t classes[simd_symbols];
    unpack_bytes(m_classes + (batch * m_width) / 8, classes);
    __m256i la, lb;
    const uint64_t sum = lens_of(classes, lanes, la, lb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens), la);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens + 32), lb);
    return sum;
  }
#else
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    uint64_t bits = 0;
    for (uint64_t x = 0; x < count; ++x) bits += m_len[m_codes[first + x]];
    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const { return prefix_bits(i, len); }

  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    uint64_t sum = 0;

    for (uint64_t x = 0; x < simd_symbols; ++x) {
      lens[x] = (lanes >> x) & 1 ? m_len[m_codes[batch + x]] : 0;
      sum += lens[x];
    }

    return sum;
  }
#endif

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_forward(uint64_t i, uint64_t d) const {
    alignas(64) uint8_t lens[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t skip = i - batch;
    uint64_t m = 0;

    for (;;) {
      const uint64_t sum = batch_lens(batch, ~uint64_t{0} << skip, lens);

      if (sum > d) {
        for (uint64_t x = skip;; ++x) {
          if (lens[x] > d) return m;
          d -= lens[x];
          ++m;
        }
      }

      d -= sum;
      m += simd_symbols - skip;
      batch += simd_symbols;
      skip = 0;
    }
  }

  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_backward(uint64_t i, uint64_t s) const {
    alignas(64) uint8_t lens[simd_symbols];
    uint64_t batch = i / simd_symbols * simd_symbols;
    uint64_t last = i - batch;
    uint64_t m = 0;

    for (;;) {
      const uint64_t sum = batch_lens(batch, last == 63 ? ~uint64_t{0} : (uint64_t{1} << (last + 1)) - 1, lens);

      if (sum > s) {
        for (uint64_t x = last;; --x) {
          if (lens[x] > s) return m;
          s -= lens[x];
          ++m;
        }
      }

      s -= sum;
      m += last + 1;
      batch -= simd_symbols;
      last = simd_symbols - 1;
    }
  }

  uint64_t common_prefix(uint64_t a, uint64_t b, uint64_t bits) const {
    uint64_t k = 0;

    while (k + window_bits <= bits) {
      const uint64_t diff = (load_bits(a + k) ^ load_bits(b + k)) & window_mask;
      if (diff != 0) return k + std::countr_zero(diff);
      k += window_bits;
    }

    if (k < bits) {
      const uint64_t diff = (load_bits(a + k) ^ load_bits(b + k)) & ((uint64_t{1} << (bits - k)) - 1);
      if (diff != 0) return k + std::countr_zero(diff);
    }

    return bits;
  }

  uint64_t common_suffix(uint64_t a, uint64_t b, uint64_t bits) const {
    uint64_t k = 0;

    while (k + window_bits <= bits) {
      const uint64_t diff = (load_bits(a - k - window_bits) ^ load_bits(b - k - window_bits)) & window_mask;
      if (diff != 0) return k + std::countl_zero(diff) - (64 - window_bits);
      k += window_bits;
    }

    if (k < bits) {
      const uint64_t rest = bits - k;
      const uint64_t diff = (load_bits(a - bits) ^ load_bits(b - bits)) & ((uint64_t{1} << rest) - 1);
      if (diff != 0) return k + std::countl_zero(diff) - (64 - rest);
    }

    return bits;
  }

  uint64_t load_bits(uint64_t bit) const { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); }

  packed_text<> m_codes;
  std::shared_ptr<data_t> m_data;
  const uint8_t* m_classes = nullptr;
  const uint8_t* m_classes_limit = nullptr;
  const uint8_t* m_payload = nullptr;
  const uint64_t* m_abs = nullptr;
  const uint8_t* m_rel = nullptr;
  const uint8_t* m_len = nullptr;
  const uint64_t* m_mask = nullptr;
  const uint32_t* m_start = nullptr;
  const uint32_t* m_value_of = nullptr;
  const uint8_t* m_spread = nullptr;
  const uint8_t* m_shifts = nullptr;
  const uint8_t* m_bit_to_symbol = nullptr;
  uint64_t m_size = 0;
  uint64_t m_payload_bits = 0;
  uint64_t m_penalty = payload_penalty;
  uint64_t m_sigma = 0;
  uint64_t m_short_mask = 0;
  uint64_t m_field_one = 0;
  uint64_t m_field_high = 0;
  uint64_t m_field_low = 0;
  uint64_t m_direct_low = 0;
  uint32_t m_avg_x256 = 0;
  uint16_t m_direct_classes = 0;
  uint8_t m_width = 0;
  uint8_t m_class_mask = 0;
  uint8_t m_short = 0;
  bool m_direct_high = false;
  bool m_eager = false;
  bool m_has_payloads = false;
};

}

#endif
