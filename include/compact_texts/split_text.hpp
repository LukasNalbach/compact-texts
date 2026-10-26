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
#include <istream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <ostream>
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
#include <misc/serialize.hpp>

#include <compact_texts/common.hpp>
#include <compact_texts/packed_text.hpp>

namespace compact_texts {

/**
 * @brief a text with a variable number of bits per symbol: each position stores the code of its class with a
 * fixed number of bits and, if its class has more than one symbol, the index of its symbol within the class
 * @tparam t_symbol uint8_t (byte text) or uint32_t (integer text)
 */
template <typename t_symbol = uint8_t>
class split_text;

/**
 * @brief a byte text in the split encoding
 *
 * The occurring characters are mapped to the symbols 0, ..., sigma - 1 in the order of the characters. The symbols
 * are partitioned into at most 2^w classes (w = width()). Class c has a payload length len_c in [0..8] and contains
 * at most 2^len_c symbols; the classes are numbered by increasing len_c, and the symbols are assigned to them in the
 * order of decreasing frequency. Any number of classes can have a payload. Symbol s of class c with index p within
 * c is stored as:
 *
 * 1. class codes: c with w bits, in a packed_text (position i at bit i * w)
 * 2. payloads: p with len_c bits, if len_c > 0; the payloads of all positions are concatenated in the order of the
 *    positions into one bit stream
 * 3. directory (only if some len_c > 0): the payload offset of the first position of each super block of
 *    super_symbols positions (64 bits) and of each block of block_symbols positions relative to its super block
 *    (offset_bytes bytes)
 *
 * The payload offset of position i is the offset of its block plus the sum of len_c over the classes c of the
 * positions of the block before i, which is computed from the class codes with vector instructions. If T[i..i+l)
 * and T[j..j+l) have the same class codes, their payloads have the same length and the same field boundaries, hence
 * lce() compares the class codes and then the payload bits, 64 bits at a time.
 */
template <>
class split_text<uint8_t> {
 public:
  using char_type = char;
  using symbol_type = uint8_t;
  using histogram_t = packed_text<>::histogram_t;

  static constexpr bool is_byte_text = true;
  // default cost in bits that the choice of the code adds for each position with a payload
  static constexpr uint64_t payload_penalty = 2;
  // number of positions decoded at once (one byte per position in a 64-byte vector)
  static constexpr uint64_t batch_symbols = 64;
  static constexpr uint64_t batches_per_block = 4;
  // number of positions per block offset
  static constexpr uint64_t block_symbols = batches_per_block * batch_symbols;
  // the parts passed to pack() must start at multiples of chunk_symbols
  static constexpr uint64_t chunk_symbols = block_symbols;
  // number of positions per super block offset
  static constexpr uint64_t super_symbols = uint64_t{1} << 20;
  static constexpr uint64_t blocks_per_super = super_symbols / block_symbols;
  // bytes per block offset
  static constexpr uint64_t offset_bytes = 3;
  // maximum payload length in bits
  static constexpr uint64_t max_payload = 8 * sizeof(symbol_type);
  // maximum class width of a code with payloads (a table with one byte per class fits into one batch-sized vector)
  static constexpr uint64_t max_split_width = std::countr_zero(batch_symbols);
  // number of zero bytes after the payload stream
  static constexpr uint64_t padding_bytes = 128;
  // ranges of at most max_summed_symbols positions get their number of payload bits by summing the payload lengths of
  // their classes, longer ranges by subtracting two payload offsets
  static constexpr uint64_t max_summed_symbols = 192;
  // number of payload bits compared per step: the whole bytes of a 64-bit word read at the byte of the first bit,
  // after shifting out the bits before the first bit
  static constexpr uint64_t window_bits = 8 * (sizeof(uint64_t) - 1);
  static constexpr uint64_t window_mask = (uint64_t{1} << window_bits) - 1;
  // number of cache lines of the payload stream prefetched before a comparison of long ranges
  static constexpr int64_t prefetched_lines = 3;
  // comparisons of long ranges prefetch the payload stream only if the average number of payload bits per position,
  // times 256, is at least min_prefetch_avg_x256
  static constexpr uint32_t min_prefetch_avg_x256 = 8;
  // lce() prefetches the payload stream at its first step if at least 1 / eager_prefetch_divisor of the positions
  // have a payload
  static constexpr uint64_t eager_prefetch_divisor = 5;

  // the payload bits of a super block fit into a block offset
  static_assert(super_symbols * max_payload <= uint64_t{1} << (8 * offset_bytes));

  // decodes the symbols at i, i + 1, ... one batch at a time
  class cursor {
   public:
    cursor() = default;

    cursor(const split_text* text, uint64_t i)
        : m_text(text), m_batch(i / batch_symbols * batch_symbols), m_at(uint32_t(i - m_batch)) {
      m_bit = text->m_has_payloads ? text->payload_bit(i) : 0;
      text->template decode_batch<false>(m_batch, m_at, batch_symbols, m_bit, m_buffer);
    }

    uint8_t next() {
      if (m_at == batch_symbols) [[unlikely]] {
        m_batch += batch_symbols;
        m_at = 0;
        m_text->template decode_batch<false>(m_batch, 0, batch_symbols, m_bit, m_buffer);
      }

      return m_buffer[m_at++];
    }

   private:
    alignas(64) uint8_t m_buffer[batch_symbols]; // the decoded symbols of the batch starting at m_batch
    const split_text* m_text = nullptr;
    uint64_t m_batch = 0; // first position of the decoded batch
    uint64_t m_bit = 0; // payload offset of the first position after the decoded batch
    uint32_t m_at = 0; // index in m_buffer of the next symbol
  };

  split_text() = default;

  /**
   * @brief builds the split text of a byte text
   * @param data the text
   * @param size its length n
   * @param threads number of threads
   */
  split_text(const char* data, uint64_t size, int threads = omp_get_max_threads())
      : split_text(packed_text<>::count_chars(data, size, threads), size) {
    pack(data, size, 0, threads);
    finish(threads);
  }

  /**
   * @brief chooses the code for a text with the given histogram and allocates the class codes; the text is then
   * stored with pack() and finish()
   * @param histogram [0..255] number of occurrences of each character
   * @param size length n of the text
   * @param penalty cost in bits added for each position with a payload when choosing the code
   */
  split_text(const histogram_t& histogram, uint64_t size, uint64_t penalty = payload_penalty)
      : m_size(size), m_penalty(penalty) {
    m_data = std::make_shared<data_t>();
    data_t& d = *m_data;
    std::array<uint64_t, 256> freq{}; // freq[s] = number of occurrences of symbol s
    std::array<uint8_t, 256> order{}; // the symbols in the order of decreasing frequency
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
    std::vector<uint64_t> sorted(sigma); // sorted[r] = frequency of the symbol with the r-th largest frequency
    for (uint64_t r = 0; r < sigma; ++r) sorted[r] = freq[order[r]];
    std::vector<uint8_t> lengths; // lengths[c] = payload length of class c, increasing
    m_width = choose_code(sorted, size, penalty, lengths);
    const uint64_t classes = uint64_t{1} << m_width;
    d.symbol_of.assign(classes << max_payload, 0);
    d.char_of.assign(classes << max_payload, 0);
    uint64_t next = 0;
    uint64_t payload_positions = 0;
    m_direct_classes = 0;

    // class c gets the next 2^lengths[c] symbols in the order of decreasing frequency (the last class can get fewer)
    for (uint64_t c = 0; c < lengths.size(); ++c) {
      const uint64_t slots = uint64_t{1} << lengths[c];
      d.payload_len[c] = lengths[c];
      d.payload_mask[c] = uint8_t(slots - 1);
      m_has_payloads = m_has_payloads || lengths[c] != 0;
      if (lengths[c] == 0) m_direct_classes = uint16_t(c + 1);

      for (uint64_t p = 0; p < slots && next < sigma; ++p, ++next) {
        const uint8_t symbol = order[next];
        d.class_of[symbol] = uint8_t(c);
        d.payload_of[symbol] = uint8_t(p);
        d.symbol_of[(c << max_payload) | p] = symbol;
        d.char_of[(c << max_payload) | p] = d.to_char[symbol];
        if (lengths[c] != 0) payload_positions += freq[symbol];
      }
    }

    m_eager_prefetch = size != 0 && payload_positions * eager_prefetch_divisor >= size;

    for (uint16_t c = 0; c < 256; ++c) {
      if (histogram[c] == 0) continue;
      const uint8_t symbol = d.to_symbol[c];
      d.class_of_char[c] = d.class_of[symbol];
      d.payload_of_char[c] = d.payload_of[symbol];
      d.payload_len_of_char[c] = d.payload_len[d.class_of[symbol]];
    }

    init_unpack_tables();

    for (uint64_t k = 0; k < 64; ++k) {
      d.direct_symbol[k] = k < classes ? d.symbol_of[k << max_payload] : 0;
      d.direct_char[k] = k < classes ? d.char_of[k << max_payload] : 0;
      d.bit_to_symbol[k] = uint8_t(k / m_width);
    }

    m_class_mask = uint8_t(classes - 1);
    m_window_codes = uint8_t(std::min<uint64_t>(window_bits / m_width, batch_symbols - 1));
    m_window_codes_mask = (uint64_t{1} << (m_window_codes * m_width)) - 1;

    // the masks of the lowest and of the highest bit of each w-bit field of a 64-bit word
    for (uint64_t field = 0; field * m_width + m_width <= 64; ++field) {
      m_field_one |= uint64_t{1} << (field * m_width);
      m_field_high |= uint64_t{1} << (field * m_width + m_width - 1);
    }

    // m_direct_classes (at most 2^w - 1), split into its highest bit and its lower w - 1 bits in each field
    m_field_low = m_field_high - m_field_one;
    const uint64_t direct = std::min<uint64_t>(m_direct_classes, classes - 1);
    m_direct_high = ((direct >> (m_width - 1)) & 1) != 0;
    m_direct_low = m_field_one * (direct & ((uint64_t{1} << (m_width - 1)) - 1));
    histogram_t class_histogram{};
    for (uint64_t c = 0; c < classes; ++c) class_histogram[c] = 1;
    m_class_codes = packed_text<>(class_histogram, size);

    if (m_has_payloads) {
      d.super_offsets.assign(size / super_symbols + 2, 0);
      d.block_offsets.assign(offset_bytes * (size / block_symbols + 2) + 8, 0);
    }

    d.payload.assign(padding_bytes, 0);
    cache_pointers();
  }

  /**
   * @brief stores the part T[at..at+length) of the text; parts can be packed by several threads at once
   * @param data the part T[at..at+length)
   * @param length its length
   * @param at its starting position in the text, a multiple of chunk_symbols
   * @param threads number of threads
   */
  void pack(const char* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) {
    const uint8_t* class_of_char = m_data->class_of_char.data();
    m_class_codes.pack_symbols(length, at, [class_of_char, data](uint64_t i) { return uint64_t(class_of_char[uint8_t(data[i])]); }, threads);
    if (!m_has_payloads || length == 0) return;
    const uint8_t* payload_of_char = m_data->payload_of_char.data();
    const uint8_t* payload_len_of_char = m_data->payload_len_of_char.data();
    const uint64_t blocks = (length + block_symbols - 1) / block_symbols;
    const uint64_t parts = std::min<uint64_t>(blocks, std::max(threads, 1));

    // each thread writes the payloads of a range of whole blocks into a piece; finish() concatenates the pieces
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
        for (uint64_t i = b * block_symbols; i < block_end; ++i) bits += payload_len_of_char[uint8_t(data[i])];
        piece.block_bits[b - first_block] = uint16_t(bits);
        total += bits;
      }

      piece.bits = total;
      resize_no_init(piece.words, (total + 63) / 64 + 1);
      uint64_t word = 0;
      uint64_t acc = 0; // the payload bits that do not fill a word yet
      uint64_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint8_t c = uint8_t(data[i]);
        const uint64_t value = payload_of_char[c];
        const uint64_t len = payload_len_of_char[c];
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

  /**
   * @brief builds the payload stream and the directory from the pieces written by pack(); called once after all
   * parts have been packed
   * @param threads number of threads
   */
  void finish(int threads = omp_get_max_threads()) {
    data_t& d = *m_data;

    if (!m_has_payloads) {
      d.pieces = decltype(d.pieces)();
      cache_pointers();
      return;
    }

    std::sort(d.pieces.begin(), d.pieces.end(), [](const piece_t& a, const piece_t& b) { return a.at < b.at; });
    const uint64_t pieces = d.pieces.size();
    std::vector<uint64_t> offset(pieces + 1, 0); // offset[p] = payload offset of the first position of piece p
    for (uint64_t p = 0; p < pieces; ++p) offset[p + 1] = offset[p] + d.pieces[p].bits;
    const uint64_t total = offset[pieces];
    m_payload_bits = total;
    m_avg_payload_bits_x256 = uint32_t(m_size == 0 ? 0 : (total * 256 + m_size / 2) / m_size);
    resize_no_init(d.payload, (total + 63) / 64 * 8 + padding_bytes);
    advise_huge_pages(d.payload.data(), d.payload.size());
    parallel_memset(d.payload.data(), 0, d.payload.size(), threads);

    // super block offsets
#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        if (block % blocks_per_super == 0) d.super_offsets[block / blocks_per_super] = cum;
        cum += piece.block_bits[k];
      }
    }

    const uint64_t end_block = (m_size + block_symbols - 1) / block_symbols;
    if (end_block % blocks_per_super == 0) d.super_offsets[end_block / blocks_per_super] = total;

    // block offsets, relative to the offsets of their super blocks
#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        write_offset(d.block_offsets.data(), block, cum - d.super_offsets[block / blocks_per_super]);
        cum += piece.block_bits[k];
      }
    }

    write_offset(d.block_offsets.data(), end_block, total - d.super_offsets[end_block / blocks_per_super]);
    uint64_t* out = reinterpret_cast<uint64_t*>(d.payload.data());

    // copies the bits of each piece to its offset; the first and the last word of a piece can be shared with the
    // neighbouring pieces and are combined with an atomic or
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
  // number w of bits per class code
  uint8_t width() const { return m_width; }
  uint64_t penalty() const { return m_penalty; }
  bool has_payloads() const { return m_has_payloads; }
  // total number of payload bits
  uint64_t payload_bits() const { return m_payload_bits; }
  // [0..2^w-1] payload length of each class
  const uint8_t* class_lengths() const { return m_payload_len; }
  // [0..255] class of each character
  const uint8_t* class_of_chars() const { return m_data->class_of_char.data(); }

  uint64_t size_in_bytes() const {
    uint64_t bytes = sizeof(*this) + m_class_codes.size_in_bytes();
    if (m_data) {
      bytes += sizeof(data_t) + m_data->payload.size() + m_data->super_offsets.size() * 8 + m_data->block_offsets.size() +
               m_data->symbol_of.size() + m_data->char_of.size();
    }
    return bytes;
  }

  // writes the text as it is in memory: its parameters, its tables, the payloads and the codes of the classes
  void serialize(std::ostream& out) const {
    if (!m_data) {
      split_text empty(histogram_t{}, 0);
      empty.finish(1);
      empty.serialize(out);
      return;
    }

    for_each_member(*this, *m_data, [&](const auto& member) {
      out.write(reinterpret_cast<const char*>(&member), sizeof(member));
    });

    const data_t& d = *m_data;
    write_vector(out, d.payload);
    write_vector(out, d.super_offsets);
    write_vector(out, d.block_offsets);
    write_vector(out, d.symbol_of);
    write_vector(out, d.char_of);
    m_class_codes.serialize(out);
  }

  // reads a text written by serialize()
  void load(std::istream& in) {
    m_data = std::make_shared<data_t>();
    data_t& d = *m_data;
    for_each_member(*this, d, [&](auto& member) { in.read(reinterpret_cast<char*>(&member), sizeof(member)); });
    read_vector(in, d.payload);
    read_vector(in, d.super_offsets);
    read_vector(in, d.block_offsets);
    read_vector(in, d.symbol_of);
    read_vector(in, d.char_of);
    m_class_codes.load(in);
    init_unpack_tables();
    cache_pointers();
  }

  /**
   * @brief returns size_in_bytes() of the split text of a text with the given histogram, without building it
   * @param histogram [0..255] number of occurrences of each character
   * @param size length n of the text
   * @param penalty cost in bits added for each position with a payload when choosing the code
   */
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

    // the class codes, the tables and the tables symbol_of and char_of
    uint64_t bytes = sizeof(split_text) + sizeof(packed_text<>) + (size * width + 63) / 64 * 8 + packed_text<>::padding_bytes +
                     sizeof(data_t) + 2 * ((uint64_t{1} << width) << max_payload);

    // the payload stream and the directory
    if (!any_payload) return bytes + padding_bytes;
    return bytes + (payload_bits + 63) / 64 * 8 + padding_bytes + (size / super_symbols + 2) * 8 +
           offset_bytes * (size / block_symbols + 2) + 8;
  }

  // returns the symbol at position i
  uint8_t operator[](uint64_t i) const {
    const uint8_t c = m_class_codes[i];
    if (m_payload_len[c] == 0) return m_symbol_of[uint64_t(c) << max_payload];
    return payload_symbol<false>(i, c);
  }

  // returns the character at position i
  char_type char_at(uint64_t i) const {
    const uint8_t c = m_class_codes[i];
    if (m_payload_len[c] == 0) return char_type(m_char_of[uint64_t(c) << max_payload]);
    return char_type(payload_symbol<true>(i, c));
  }

  // returns whether T[i] < T[j]
  bool less_char(uint64_t i, uint64_t j) const { return (*this)[i] < (*this)[j]; }
  // returns the character of a symbol
  uint8_t to_char(uint8_t symbol) const { return m_to_char[symbol]; }
  // returns the symbol of a character
  uint8_t to_symbol(uint8_t c) const { return m_to_symbol[c]; }
  // returns a cursor whose next() returns T[i], T[i + 1], ...
  cursor cursor_at(uint64_t i) const { return cursor(this, i); }

  /**
   * @brief writes the characters T[i..i+len) to out
   * @param i [0..n-1] position
   * @param len number of characters
   * @param out array of at least len bytes
   */
  void extract(uint64_t i, uint64_t len, uint8_t* __restrict out) const {
    if (len == 0) return;
    if (m_has_payloads) prefetch_payload(i);
    uint64_t bit = m_has_payloads ? payload_bit(i) : 0;
    alignas(64) uint8_t buffer[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
    uint64_t skip = i - batch;
    uint64_t k = 0;

    while (k < len) {
      const uint64_t take = std::min(len - k, batch_symbols - skip);
      decode_batch<true>(batch, skip, skip + take, bit, buffer);
      std::memcpy(out + k, buffer + skip, take);
      k += take;
      batch += batch_symbols;
      skip = 0;
    }
  }

  /**
   * @brief returns the length of the longest common prefix of T[i..n) and T[j..n), at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_class_codes.lce(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t limit = std::min(max, m_size - std::max(i, j));
    if (limit == 0) [[unlikely]] return 0;

    // compare the class codes of the first m_window_codes positions
    const uint64_t wi = class_window(i);
    const uint64_t wj = class_window(j);
    const uint64_t diff = wi ^ wj;
    if ((diff & m_class_mask) != 0) return 0;

    if (m_eager_prefetch) {
      prefetch_predicted(i, diff == 0 ? prefetched_lines : 1);
      prefetch_predicted(j, diff == 0 ? prefetched_lines : 1);
    }

    // the class codes differ within the window: compare the payloads of the positions before the difference
    if (diff != 0 || limit <= m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_bit_to_symbol[std::countr_zero(diff)] : m_window_codes);
      if (!window_has_payload(wi, len)) return len;
      return payload_lce(i, j, len);
    }

    if (!m_eager_prefetch) {
      prefetch_block_offset(i);
      prefetch_block_offset(j);
    }

    // the same for the next m_window_codes positions
    const uint64_t vi = class_window(i + m_window_codes);
    const uint64_t vj = class_window(j + m_window_codes);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, m_window_codes + (diff2 != 0 ? m_bit_to_symbol[std::countr_zero(diff2)] : m_window_codes));
      if (!window_has_payload(wi, m_window_codes) && !window_has_payload(vi, len - m_window_codes)) return len;
      return payload_lce(i, j, len);
    }

    return long_lce(i, j, limit);
  }

  /**
   * @brief returns the length of the longest common suffix of T[0..i] and T[0..j], at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce_left(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_class_codes.lce_left(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min(max, std::min(i, j) + 1);
    if (limit == 0) [[unlikely]] return 0;
    if (std::min(i, j) + 1 < 2 * m_window_codes) [[unlikely]] return lce_left_near_start(i, j, limit);

    // as lce(), with windows that end at i and j
    const uint64_t wi = class_window(i + 1 - m_window_codes);
    const uint64_t wj = class_window(j + 1 - m_window_codes);
    const uint64_t diff = wi ^ wj;
    if ((diff >> ((m_window_codes - 1) * m_width)) != 0) return 0;

    if (m_eager_prefetch) {
      prefetch_predicted(i, diff == 0 ? -prefetched_lines : -1);
      prefetch_predicted(j, diff == 0 ? -prefetched_lines : -1);
    }

    if (diff != 0 || limit <= m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_window_codes - 1 - m_bit_to_symbol[63 - std::countl_zero(diff)] : m_window_codes);
      if (!window_has_payload(wi >> ((m_window_codes - len) * m_width), len)) return len;
      return payload_lce_left(i, j, len);
    }

    if (!m_eager_prefetch) {
      prefetch_block_offset(i);
      prefetch_block_offset(j);
    }

    const uint64_t vi = class_window(i + 1 - 2 * m_window_codes);
    const uint64_t vj = class_window(j + 1 - 2 * m_window_codes);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, m_window_codes + (diff2 != 0 ? m_window_codes - 1 - m_bit_to_symbol[63 - std::countl_zero(diff2)] : m_window_codes));
      const uint64_t rest = len - m_window_codes;
      if (!window_has_payload(wi, m_window_codes) && !window_has_payload(vi >> ((m_window_codes - rest) * m_width), rest)) return len;
      return payload_lce_left(i, j, len);
    }

    return long_lce_left(i, j, limit);
  }

  // returns whether T[i..i+len) = T[j..j+len)
  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  // returns a hash value of T[i..i+len), computed from the class codes and the payload bits of the range
  uint64_t hash(uint64_t i, uint64_t len) const {
    const uint64_t h = m_class_codes.hash(i, len);
    if (!m_has_payloads || len == 0) return h;
    return payload_hash(i, len, h);
  }

 private:
  // the payloads of a range of whole blocks, written by one thread in pack()
  struct piece_t {
    uint64_t at = 0; // first position of the range
    uint64_t bits = 0; // number of payload bits
    no_init_vector<uint64_t> words; // the payload bits
    std::vector<uint16_t> block_bits; // block_bits[k] = number of payload bits of the k-th block of the range
  };

  // the arrays and tables of the text (shared by the copies of a text)
  struct data_t {
    no_init_vector<uint8_t> payload; // the payload stream, followed by padding_bytes zero bytes
    std::vector<uint64_t> super_offsets; // [0..n/super_symbols+1] payload offset of the first position of each super block
    std::vector<uint8_t> block_offsets; // [0..n/block_symbols+1] the same for each block, relative to its super block
    std::vector<uint8_t> symbol_of; // [(c << max_payload) | p] symbol with index p in class c
    std::vector<uint8_t> char_of; // [(c << max_payload) | p] character with index p in class c
    std::vector<piece_t> pieces; // the pieces written by pack() until finish()
    std::mutex mutex; // guards pieces
    alignas(64) std::array<uint8_t, 256> payload_len{}; // [0..2^w-1] payload length of class c
    alignas(64) std::array<uint8_t, 256> payload_mask{}; // [0..2^w-1] 2^payload_len[c] - 1
    alignas(64) std::array<uint8_t, 64> spread{}; // byte indices that unpack the class codes (see init_unpack_tables)
    alignas(64) std::array<uint8_t, 64> shifts{}; // bit shifts or multipliers that unpack the class codes
    alignas(64) std::array<uint8_t, 64> direct_symbol{}; // [0..63] symbol with index 0 in class c
    alignas(64) std::array<uint8_t, 64> direct_char{}; // [0..63] character with index 0 in class c
    alignas(64) std::array<uint8_t, 64> bit_to_symbol{}; // [0..63] index of the class code that contains bit k of a window
    std::array<uint8_t, 256> class_of{}; // [0..sigma-1] class of symbol s
    std::array<uint8_t, 256> payload_of{}; // [0..sigma-1] index of symbol s in its class
    std::array<uint8_t, 256> class_of_char{}; // [0..255] class of character c
    std::array<uint8_t, 256> payload_of_char{}; // [0..255] index of character c in its class
    std::array<uint8_t, 256> payload_len_of_char{}; // [0..255] payload length of character c
    std::array<uint8_t, 256> to_symbol{}; // [0..255] symbol of character c
    std::array<uint8_t, 256> to_char{}; // [0..sigma-1] character of symbol s
  };

  // calls io(x) for each member x of the text t and of its tables d that serialize() writes as it is
  template <typename text_t, typename tables_t, typename io_t>
  static void for_each_member(text_t& t, tables_t& d, io_t io) {
    io(t.m_size);
    io(t.m_payload_bits);
    io(t.m_penalty);
    io(t.m_window_codes_mask);
    io(t.m_field_one);
    io(t.m_field_high);
    io(t.m_field_low);
    io(t.m_direct_low);
    io(t.m_avg_payload_bits_x256);
    io(t.m_sigma);
    io(t.m_direct_classes);
    io(t.m_width);
    io(t.m_class_mask);
    io(t.m_window_codes);
    io(t.m_direct_high);
    io(t.m_eager_prefetch);
    io(t.m_has_payloads);
    io(d.payload_len);
    io(d.payload_mask);
    io(d.direct_symbol);
    io(d.direct_char);
    io(d.bit_to_symbol);
    io(d.class_of);
    io(d.payload_of);
    io(d.class_of_char);
    io(d.payload_of_char);
    io(d.payload_len_of_char);
    io(d.to_symbol);
    io(d.to_char);
  }

  /**
   * @brief computes the tables spread and shifts that unpack the class codes of a batch into one byte per position
   *
   * AVX-512 VBMI: byte k of the vector is byte spread[k] = l * w + (k mod 8) of the class codes of the batch, where
   * l = k / 8, so the 64-bit lane l starts with the class codes 8l, ..., 8l + 7; class code k is then the lowest w of
   * the 8 bits at bit shifts[k] = (k mod 8) * w of its lane.
   * AVX2: each 128-bit half of a vector covers 8 class codes; the 16-bit element e of a half consists of the bytes
   * spread[2e] and spread[2e + 1], which contain class code e; multiplying it by shifts[2e] + 256 * shifts[2e + 1] =
   * 2^(8 - b), where b is the bit offset of class code e in its first byte, moves the class code to the high byte.
   */
  void init_unpack_tables() {
    data_t& d = *m_data;

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
    }
  }

  // sets the pointers into the tables and arrays
  void cache_pointers() {
    data_t& d = *m_data;
    m_class_code_data = m_class_codes.packed_data();
    m_class_code_data_end = m_class_code_data + (m_size * m_width + 63) / 64 * 8;
    m_payload = d.payload.data();
    m_super_offsets = d.super_offsets.data();
    m_block_offsets = d.block_offsets.data();
    m_payload_len = d.payload_len.data();
    m_payload_mask = d.payload_mask.data();
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

  // returns the number of bits of the directory of a text of length size
  static uint64_t directory_bits(uint64_t size) {
    return (size / super_symbols + 2) * 64 + (size / block_symbols + 2) * 8 * offset_bytes;
  }

  /**
   * @brief chooses the code with the smallest cost (bits of the class codes and the payloads, penalty bits per
   * position with a payload, bits of the directory); widths larger than max_split_width are only possible with one
   * class per symbol
   * @param sorted the frequencies of the symbols in decreasing order
   * @param size length n of the text
   * @param penalty cost in bits added for each position with a payload
   * @param lengths receives the payload lengths of the classes, increasing
   * @return the class width w
   */
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
        if (any_payload) cost += double(directory_bits(size));
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

  /**
   * @brief computes the payload lengths of the classes of the cheapest code with class width w by dynamic
   * programming in O(2^w * sigma * max_payload) time
   *
   * f[c][p] is the smallest cost of encoding the p most frequent symbols with c classes, where a class with payload
   * length l takes the next 2^l symbols (the last class can take fewer) and costs w + l (+ penalty, if l > 0) bits per
   * occurrence of its symbols.
   * @param sorted the frequencies of the symbols in decreasing order
   * @param width class width w
   * @param penalty cost in bits added for each position with a payload
   * @param lengths receives the payload lengths of the classes, increasing
   * @return the cost in bits, or infinity if 2^w classes cannot hold all symbols
   */
  static double optimal_lengths(const std::vector<uint64_t>& sorted, uint8_t width, uint64_t penalty,
                                std::vector<uint8_t>& lengths) {
    const uint64_t m = sorted.size();
    const uint64_t classes = uint64_t{1} << width;
    const double inf = std::numeric_limits<double>::max();
    std::vector<double> prefix(m + 1, 0); // prefix[p] = number of occurrences of the p most frequent symbols
    for (uint64_t p = 0; p < m; ++p) prefix[p + 1] = prefix[p] + double(sorted[p]);
    std::vector<double> f((classes + 1) * (m + 1), inf);
    std::vector<uint8_t> from_len((classes + 1) * (m + 1), 0); // payload length of the last class of f[c][p]
    std::vector<uint16_t> from_p((classes + 1) * (m + 1), 0); // number of symbols before the last class of f[c][p]
    auto at = [&](uint64_t c, uint64_t p) { return c * (m + 1) + p; };
    f[at(0, 0)] = 0;

    for (uint8_t l = 0; l <= max_payload; ++l) {
      const uint64_t slots = uint64_t{1} << l;
      const double per_symbol = double(width) + double(l) + (l != 0 ? double(penalty) : 0.0);

      for (uint64_t c = 1; c <= classes; ++c) {
        // the c-th class is full
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

        // the c-th class is the last one and takes the remaining fewer than 2^l symbols
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

    // the classes in the order of increasing payload length, and the cost of assigning the symbols in this order
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

  // writes the relative offset of a block (offset_bytes bytes, little endian)
  static void write_offset(uint8_t* block_offsets, uint64_t block, uint64_t value) {
    for (uint64_t k = 0; k < offset_bytes; ++k) block_offsets[offset_bytes * block + k] = uint8_t(value >> (8 * k));
  }

  // returns the payload offset of the first position of a block
  uint64_t block_bit(uint64_t block) const {
    uint32_t value;
    std::memcpy(&value, m_block_offsets + offset_bytes * block, sizeof(value));
    return m_super_offsets[block / blocks_per_super] + (value & ((uint32_t{1} << (8 * offset_bytes)) - 1));
  }

  // returns the payload offset of position i: the offset of its block plus the payload lengths of the positions of
  // the block before i
  uint64_t payload_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + prefix_bits(block * block_symbols, i - block * block_symbols);
  }

  // returns an estimate of the payload offset of position i from the offset of its block and the average number of
  // payload bits per position
  uint64_t predicted_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + (((i - block * block_symbols) * m_avg_payload_bits_x256) >> 8);
  }

  void prefetch_block_offset(uint64_t i) const { __builtin_prefetch(m_block_offsets + offset_bytes * (i / block_symbols)); }

  // prefetches the cache line of the estimated payload offset of position i
  void prefetch_payload(uint64_t i) const { __builtin_prefetch(m_payload + (predicted_bit(i) >> 3)); }

  // prefetches the cache lines of the payload bits [bit, bit + bits)
  void prefetch_range(uint64_t bit, uint64_t bits) const {
    const uint8_t* end = m_payload + ((bit + bits) >> 3);
    for (const uint8_t* p = m_payload + (bit >> 3); p <= end; p += 64) __builtin_prefetch(p);
  }

  // prefetches lines cache lines from the estimated payload offset of position i on, backwards if lines < 0
  void prefetch_predicted(uint64_t i, int64_t lines) const {
    const uint64_t byte = predicted_bit(i) >> 3;

    if (lines > 0) {
      for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
    } else {
      for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
    }
  }

  // returns the symbol (t_chars = false) or the character (t_chars = true) at position i of class c > 0
  template <bool t_chars>
  uint8_t payload_symbol(uint64_t i, uint8_t c) const {
    prefetch_payload(i);
    const uint64_t bit = payload_bit(i);
    const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
    return (t_chars ? m_char_of : m_symbol_of)[(uint64_t(c) << max_payload) | payload];
  }

  // returns the class codes of the positions [i, i + m_window_codes), position i in the lowest bits
  uint64_t class_window(uint64_t i) const {
    const uint64_t bit = i * m_width;
    return (load_u64(m_class_code_data + (bit >> 3)) >> (bit & 7)) & m_window_codes_mask;
  }

  // returns whether one of the lowest len class codes in window is at least m_direct_classes, i.e., has a payload;
  // in t, the highest bit of a field is set iff the lower w - 1 bits of its class code are at least those of
  // m_direct_classes
  bool window_has_payload(uint64_t window, uint64_t len) const {
    const uint64_t x = window & ((uint64_t{1} << (len * m_width)) - 1);
    const uint64_t t = ((x & m_field_low) | m_field_high) - m_direct_low;
    const uint64_t r = (m_direct_high ? (x & t) : (x | t)) & m_field_high;
    return r != 0;
  }

  // returns the lce of T[i..] and T[j..], given that their first len class codes are equal and that it is at most len
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t pi = payload_bit(i);
    const uint64_t bits = len <= max_summed_symbols ? range_bits(i, len) : payload_bit(i + len) - pi;
    if (bits == 0) return len;
    const uint64_t pj = payload_bit(j);
    prefetch_range(pi, bits);
    prefetch_range(pj, bits);
    const uint64_t common = common_prefix(pi, pj, bits);
    if (common == bits) return len;
    return select_forward(i, common);
  }

  // returns the lce_left of T[..i] and T[..j], given that their last len class codes are equal and that it is at most
  // len
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t first = i + 1 - len;
    const uint64_t ei = payload_bit(i + 1);
    const uint64_t bits = len <= max_summed_symbols ? range_bits(first, len) : ei - payload_bit(first);
    if (bits == 0) return len;
    const uint64_t ej = payload_bit(j + 1);
    prefetch_range(ei - bits, bits);
    prefetch_range(ej - bits, bits);
    const uint64_t common = common_suffix(ei, ej, bits);
    if (common == bits) return len;
    return select_backward(i, common);
  }

  // returns lce(i, j, limit), given that the first 2 * m_window_codes class codes are equal: computes the lce of the
  // class codes, then compares the payloads of the equal class codes
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
      prefetch_predicted(i, prefetched_lines);
      prefetch_predicted(j, prefetched_lines);
    }

    const uint64_t skip = 2 * m_window_codes;
    const uint64_t len = skip + m_class_codes.lce(i + skip, j + skip, limit - skip);
    return payload_lce(i, j, len);
  }

  // returns lce_left(i, j, limit), given that the last 2 * m_window_codes class codes are equal
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
      prefetch_predicted(i, -prefetched_lines);
      prefetch_predicted(j, -prefetched_lines);
    }

    const uint64_t skip = 2 * m_window_codes;
    const uint64_t len = skip + m_class_codes.lce_left(i - skip, j - skip, limit - skip);
    return payload_lce_left(i, j, len);
  }

  // returns lce_left(i, j, limit) for min(i, j) < 2 * m_window_codes - 1, where the windows of lce_left() would start
  // before position 0
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t lce_left_near_start(uint64_t i, uint64_t j, uint64_t limit) const {
    const uint64_t len = m_class_codes.lce_left(i, j, limit);
    if (len == 0) return 0;
    return payload_lce_left(i, j, len);
  }

  // returns h mixed with the hash value of the payload bits of T[i..i+len)
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

  // returns the mask of the lanes [skip, end) of a batch
  static uint64_t lane_range(uint64_t skip, uint64_t end) {
    return (end >= batch_symbols ? ~uint64_t{0} : (uint64_t{1} << end) - 1) & (~uint64_t{0} << skip);
  }

  /**
   * @brief decodes the positions batch + [skip, end) into out[skip..end)
   * @tparam t_chars false: symbols, true: characters
   * @param batch first position of the batch, a multiple of batch_symbols
   * @param position payload offset of position batch + skip; receives the payload offset of position batch + end
   */
  template <bool t_chars>
  void decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint8_t* __restrict out) const {
    const uint8_t* table = t_chars ? m_char_of : m_symbol_of;
    uint64_t bit = position;
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI) || defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
    if (m_width <= max_split_width) {
      // decode all positions as if they had payload 0, then correct those with a payload one by one
      alignas(64) uint8_t classes[batch_symbols];
      const uint64_t m = decode_direct(batch, t_chars ? m_direct_char : m_direct_symbol, classes, out) & lane_range(skip, end);

      if (m_has_payloads) {
        uint64_t rest = m;

        while (rest != 0) {
          const uint64_t x = uint64_t(std::countr_zero(rest));
          rest &= rest - 1;
          const uint8_t c = classes[x];
          const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
          bit += m_payload_len[c];
          out[x] = table[(uint64_t(c) << max_payload) | payload];
        }
      }

      position = bit;
      return;
    }
#endif

    packed_text<>::cursor classes = m_class_codes.cursor_at(batch + skip);

    for (uint64_t x = skip; x < end; ++x) {
      const uint8_t c = classes.next();
      const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
      bit += m_payload_len[c];
      out[x] = table[(uint64_t(c) << max_payload) | payload];
    }

    position = bit;
  }

#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
  // returns the batch_symbols class codes stored from byte p on, one byte per class code
  __m512i unpack_at(const uint8_t* p) const {
    const __m512i raw = _mm512_loadu_si512(p);
    const __m512i spread = _mm512_permutexvar_epi8(_mm512_load_si512(m_spread), raw);
    const __m512i fields = _mm512_multishift_epi64_epi8(_mm512_load_si512(m_shifts), spread);
    return _mm512_and_si512(fields, _mm512_set1_epi8(char(m_class_mask)));
  }

  // returns the class codes of the batch starting at position first (a multiple of batch_symbols)
  __m512i unpack(uint64_t first) const { return unpack_at(m_class_code_data + (first * m_width) / 8); }

  // writes the class codes of a batch to classes and direct[c] for each class code c to out; returns the mask of the
  // lanes whose class has a payload
  uint64_t decode_direct(uint64_t batch, const uint8_t* direct, uint8_t* classes, uint8_t* __restrict out) const {
    const __m512i cls = unpack(batch);
    _mm512_store_si512(classes, cls);
    _mm512_storeu_si512(out, _mm512_permutexvar_epi8(cls, _mm512_load_si512(direct)));
    if (!m_has_payloads) return 0;
    const __m512i lens = _mm512_permutexvar_epi8(cls, _mm512_load_si512(m_payload_len));
    return _cvtmask64_u64(_mm512_test_epi8_mask(lens, lens));
  }

  // returns the number of payload bits of the positions [first, first + count), where first is the first position of
  // a block and count < block_symbols; the reads of the batches after the class codes start at m_class_code_data_end
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const __m512i zero = _mm512_setzero_si512();
    const uint8_t* p = m_class_code_data + (first * m_width) / 8;
    __m512i acc = zero;
    const __m512i table = _mm512_load_si512(m_payload_len);
    const uint64_t batch_bytes = batch_symbols * m_width / 8;

    for (uint64_t c = 0; c < batches_per_block; ++c) {
      const uint64_t take = std::min<uint64_t>(count > c * batch_symbols ? count - c * batch_symbols : 0, batch_symbols);
      const __mmask64 k = _cvtu64_mask64(take >= batch_symbols ? ~uint64_t{0} : (uint64_t{1} << take) - 1);
      const __m512i cls = unpack_at(std::min(p + c * batch_bytes, m_class_code_data_end));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, cls, table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  // returns the number of payload bits of the positions [i, i + len)
  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i table = _mm512_load_si512(m_payload_len);
    const uint64_t end = i + len;
    __m512i acc = zero;

    for (uint64_t batch = i / batch_symbols * batch_symbols; batch < end; batch += batch_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, batch_symbols);
      const __mmask64 k = _cvtu64_mask64(lane_range(lo, hi));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, unpack(batch), table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    const __mmask64 k = _cvtu64_mask64(lanes);
    const __m512i l = _mm512_maskz_permutexvar_epi8(k, unpack(batch), _mm512_load_si512(m_payload_len));
    _mm512_store_si512(lens, l);
    return uint64_t(_mm512_reduce_add_epi64(_mm512_sad_epu8(l, _mm512_setzero_si512())));
  }
#elif defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
  // returns the 16 class codes stored from byte p on, one per 16-bit element (garbage above the lowest w bits)
  __m256i unpack_pair(const uint8_t* p) const {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + m_width));
    const __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i pairs = _mm256_shuffle_epi8(raw, _mm256_load_si256(reinterpret_cast<const __m256i*>(m_spread)));
    const __m256i scale = _mm256_load_si256(reinterpret_cast<const __m256i*>(m_shifts));
    return _mm256_srli_epi16(_mm256_mullo_epi16(pairs, scale), 8);
  }

  // writes the batch_symbols class codes stored from byte p on to out, one byte per class code
  void unpack_bytes(const uint8_t* p, uint8_t* out) const {
    const __m256i mask = _mm256_set1_epi8(char(m_class_mask));
    const uint64_t step = 2 * m_width;
    const __m256i a = _mm256_packus_epi16(unpack_pair(p), unpack_pair(p + step));
    const __m256i b = _mm256_packus_epi16(unpack_pair(p + 2 * step), unpack_pair(p + 3 * step));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_and_si256(_mm256_permute4x64_epi64(a, 0xd8), mask));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_and_si256(_mm256_permute4x64_epi64(b, 0xd8), mask));
  }

  // returns table[c] for each class code c in classes; table has 2^w <= 64 entries, looked up 16 at a time
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

  // returns 0xFF in byte k if bit k of lanes is set, else 0
  static __m256i lane_bytes(uint32_t lanes) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(lanes)),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                                           2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
  }

  // returns the sum of the 64 bytes of a and b
  static uint64_t sum_bytes(__m256i a, __m256i b) {
    const __m256i s = _mm256_add_epi64(_mm256_sad_epu8(a, _mm256_setzero_si256()), _mm256_sad_epu8(b, _mm256_setzero_si256()));
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return uint64_t(_mm_cvtsi128_si64(t)) + uint64_t(_mm_extract_epi64(t, 1));
  }

  // writes the payload lengths of the 64 class codes in classes to la and lb (0 outside the mask lanes) and returns
  // their sum
  uint64_t lens_of(const uint8_t* classes, uint64_t lanes, __m256i& la, __m256i& lb) const {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    la = _mm256_and_si256(lookup(m_payload_len, a), lane_bytes(uint32_t(lanes)));
    lb = _mm256_and_si256(lookup(m_payload_len, b), lane_bytes(uint32_t(lanes >> 32)));
    return sum_bytes(la, lb);
  }

  // writes the class codes of a batch to classes and direct[c] for each class code c to out; returns the mask of the
  // lanes whose class has a payload
  uint64_t decode_direct(uint64_t batch, const uint8_t* direct, uint8_t* classes, uint8_t* __restrict out) const {
    unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), lookup(direct, a));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), lookup(direct, b));
    if (!m_has_payloads) return 0;
    const __m256i zero = _mm256_setzero_si256();
    const uint32_t ma = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lookup(m_payload_len, a), zero)));
    const uint32_t mb = uint32_t(_mm256_movemask_epi8(_mm256_cmpeq_epi8(lookup(m_payload_len, b), zero)));
    return ~(uint64_t(ma) | (uint64_t(mb) << 32));
  }

  // returns the number of payload bits of the positions [first, first + count), where first is the first position of
  // a block and count < block_symbols
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const uint64_t batch_bytes = batch_symbols * m_width / 8;
    const uint8_t* p = m_class_code_data + (first * m_width) / 8;
    alignas(64) uint8_t classes[batch_symbols];
    uint64_t bits = 0;

    for (uint64_t c = 0; c < batches_per_block && count > c * batch_symbols; ++c) {
      const uint64_t take = std::min<uint64_t>(count - c * batch_symbols, batch_symbols);
      unpack_bytes(std::min(p + c * batch_bytes, m_class_code_data_end), classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(0, take), la, lb);
    }

    return bits;
  }

  // returns the number of payload bits of the positions [i, i + len)
  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const uint64_t end = i + len;
    alignas(64) uint8_t classes[batch_symbols];
    uint64_t bits = 0;

    for (uint64_t batch = i / batch_symbols * batch_symbols; batch < end; batch += batch_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, batch_symbols);
      unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(lo, hi), la, lb);
    }

    return bits;
  }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    alignas(64) uint8_t classes[batch_symbols];
    unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
    __m256i la, lb;
    const uint64_t sum = lens_of(classes, lanes, la, lb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens), la);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens + 32), lb);
    return sum;
  }
#else
  // returns the number of payload bits of the positions [first, first + count)
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    uint64_t bits = 0;
    for (uint64_t x = 0; x < count; ++x) bits += m_payload_len[m_class_codes[first + x]];
    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const { return prefix_bits(i, len); }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    uint64_t sum = 0;

    for (uint64_t x = 0; x < batch_symbols; ++x) {
      lens[x] = (lanes >> x) & 1 ? m_payload_len[m_class_codes[batch + x]] : 0;
      sum += lens[x];
    }

    return sum;
  }
#endif

  // returns the number m of positions i, i + 1, ... before the position whose payload contains the payload bit d,
  // counted from the payload offset of i (positions without payload are counted)
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_forward(uint64_t i, uint64_t d) const {
    alignas(64) uint8_t lens[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
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
      m += batch_symbols - skip;
      batch += batch_symbols;
      skip = 0;
    }
  }

  // returns the number m of positions i, i - 1, ... after the position whose payload contains the s-th payload bit
  // before the end of the payload of i
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_backward(uint64_t i, uint64_t s) const {
    alignas(64) uint8_t lens[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
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
      batch -= batch_symbols;
      last = batch_symbols - 1;
    }
  }

  // returns the number of equal bits at the starts of the payload bits [a, a + bits) and [b, b + bits)
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

  // returns the number of equal bits at the ends of the payload bits [a - bits, a) and [b - bits, b)
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

  // returns the 64-bit word of the payload stream starting at bit bit (at least window_bits + 1 valid bits)
  uint64_t load_bits(uint64_t bit) const { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); }

  packed_text<> m_class_codes; // the class code of each position, w bits per position
  std::shared_ptr<data_t> m_data;
  // pointers into m_class_codes and m_data
  const uint8_t* m_class_code_data = nullptr;
  const uint8_t* m_class_code_data_end = nullptr; // end of the words of m_class_codes
  const uint8_t* m_payload = nullptr;
  const uint64_t* m_super_offsets = nullptr;
  const uint8_t* m_block_offsets = nullptr;
  const uint8_t* m_payload_len = nullptr;
  const uint8_t* m_payload_mask = nullptr;
  const uint8_t* m_symbol_of = nullptr;
  const uint8_t* m_char_of = nullptr;
  const uint8_t* m_to_symbol = nullptr;
  const uint8_t* m_to_char = nullptr;
  const uint8_t* m_spread = nullptr;
  const uint8_t* m_shifts = nullptr;
  const uint8_t* m_direct_symbol = nullptr;
  const uint8_t* m_direct_char = nullptr;
  const uint8_t* m_bit_to_symbol = nullptr;
  uint64_t m_size = 0; // n
  uint64_t m_payload_bits = 0; // total number of payload bits
  uint64_t m_penalty = payload_penalty;
  uint64_t m_window_codes_mask = 0; // mask of the lowest m_window_codes * w bits
  uint64_t m_field_one = 0; // the lowest bit of each w-bit field of a word
  uint64_t m_field_high = 0; // the highest bit of each w-bit field of a word
  uint64_t m_field_low = 0; // the lower w - 1 bits of each w-bit field of a word
  uint64_t m_direct_low = 0; // the lower w - 1 bits of m_direct_classes in each w-bit field of a word
  uint32_t m_avg_payload_bits_x256 = 0; // average number of payload bits per position, times 256
  uint16_t m_sigma = 0;
  uint16_t m_direct_classes = 0; // number of classes without payload (the classes 0, ..., m_direct_classes - 1)
  uint8_t m_width = 0; // w
  uint8_t m_class_mask = 0; // 2^w - 1
  uint8_t m_window_codes = 0; // number of class codes compared at once by lce() and lce_left()
  bool m_direct_high = false; // whether the highest bit of min(m_direct_classes, 2^w - 1) is set
  bool m_eager_prefetch = false; // whether lce() prefetches the payload stream at its first step
  bool m_has_payloads = false; // whether some class has a payload
};

/**
 * @brief an integer text in the split encoding
 *
 * The values are mapped to the symbols 0, ..., sigma - 1 in the order of decreasing frequency, so the symbols of class
 * c are the consecutive symbols first_symbol[c], first_symbol[c] + 1, ..., and the payload of a position is its symbol
 * minus first_symbol[c]. Payloads have up to max_payload bits. The layout and the queries are those of
 * split_text<uint8_t>; the code is chosen by one dynamic program over a grid of class boundaries (see code_grid).
 */
template <>
class split_text<uint32_t> {
 public:
  using char_type = uint32_t;
  using symbol_type = uint32_t;
  using histogram_t = std::vector<uint64_t>;

  static constexpr bool is_byte_text = false;
  // default cost in bits that the choice of the code adds for each position with a payload
  static constexpr uint64_t payload_penalty = 2;
  // number of positions decoded at once
  static constexpr uint64_t batch_symbols = 64;
  static constexpr uint64_t batches_per_block = 4;
  // number of positions per block offset
  static constexpr uint64_t block_symbols = batches_per_block * batch_symbols;
  // the parts passed to pack() must start at multiples of chunk_symbols
  static constexpr uint64_t chunk_symbols = block_symbols;
  // number of positions per super block offset
  static constexpr uint64_t super_symbols = uint64_t{1} << 16;
  static constexpr uint64_t blocks_per_super = super_symbols / block_symbols;
  // bytes per block offset
  static constexpr uint64_t offset_bytes = 3;
  // maximum payload length in bits
  static constexpr uint64_t max_payload = 8 * sizeof(symbol_type);
  // maximum class width of a code with payloads (a table with one byte per class fits into one batch-sized vector)
  static constexpr uint64_t max_split_width = std::countr_zero(batch_symbols);
  // maximum class width (the class codes are stored in a packed byte text)
  static constexpr uint64_t max_width = packed_text<>::max_width;
  // number of zero bytes after the payload stream
  static constexpr uint64_t padding_bytes = 128;
  // ranges of at most max_summed_symbols positions get their number of payload bits by summing the payload lengths of
  // their classes, longer ranges by subtracting two payload offsets
  static constexpr uint64_t max_summed_symbols = 192;
  // number of payload bits compared per step: the whole bytes of a 64-bit word read at the byte of the first bit,
  // after shifting out the bits before the first bit
  static constexpr uint64_t window_bits = 8 * (sizeof(uint64_t) - 1);
  static constexpr uint64_t window_mask = (uint64_t{1} << window_bits) - 1;
  // number of cache lines of the payload stream prefetched before a comparison of long ranges
  static constexpr int64_t prefetched_lines = 3;
  // comparisons of long ranges prefetch the payload stream only if the average number of payload bits per position,
  // times 256, is at least min_prefetch_avg_x256
  static constexpr uint32_t min_prefetch_avg_x256 = 8;
  // lce() prefetches the payload stream at its first step if at least 1 / eager_prefetch_divisor of the positions
  // have a payload
  static constexpr uint64_t eager_prefetch_divisor = 5;
  // the code_grid allows a class to end after each of the exact_code_symbols most frequent symbols and at about
  // coarse_code_positions evenly spaced positions after them
  static constexpr uint64_t exact_code_symbols = uint64_t{1} << 12;
  static constexpr uint64_t coarse_code_positions = uint64_t{1} << 12;
  static constexpr uint64_t max_classes = uint64_t{1} << max_split_width;
  // count_symbols() uses one histogram per thread if they have at most max_partial_histogram_entries entries in
  // total, else atomic increments of one histogram
  static constexpr uint64_t max_partial_histogram_entries = uint64_t{1} << 24;

  // the payload bits of a super block fit into a block offset
  static_assert(super_symbols * max_payload <= uint64_t{1} << (8 * offset_bytes));

  // decodes the symbols at i, i + 1, ... one batch at a time
  class cursor {
   public:
    cursor() = default;

    cursor(const split_text* text, uint64_t i)
        : m_text(text), m_batch(i / batch_symbols * batch_symbols), m_at(uint32_t(i - m_batch)) {
      m_bit = text->m_has_payloads ? text->payload_bit(i) : 0;
      text->decode_batch(m_batch, m_at, batch_symbols, m_bit, m_buffer);
    }

    uint32_t next() {
      if (m_at == batch_symbols) [[unlikely]] {
        m_batch += batch_symbols;
        m_at = 0;
        m_text->decode_batch(m_batch, 0, batch_symbols, m_bit, m_buffer);
      }

      return m_buffer[m_at++];
    }

   private:
    alignas(64) uint32_t m_buffer[batch_symbols];
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
    std::vector<uint64_t> sorted; // sorted[s] = frequency of symbol s
    resize_for_overwrite(sorted, sigma);
    for (uint64_t r = 0; r < sigma; ++r) sorted[r] = histogram[d.value_of[r]];
    std::vector<uint8_t> lengths; // lengths[c] = payload length of class c, increasing
    m_width = choose_code(sorted, size, penalty, lengths);
    const uint64_t classes = uint64_t{1} << m_width;
    resize_no_init(d.code_of, sigma);
    uint64_t next = 0;
    uint64_t payload_positions = 0;
    m_direct_classes = 0;

    // class c gets the next 2^lengths[c] symbols (the last class can get fewer)
    for (uint64_t c = 0; c < lengths.size(); ++c) {
      const uint64_t take = std::min<uint64_t>(uint64_t{1} << lengths[c], sigma - std::min(sigma, next));
      d.payload_len[c] = lengths[c];
      d.payload_mask[c] = (uint64_t{1} << lengths[c]) - 1;
      d.first_symbol[c] = uint32_t(next);
      m_has_payloads = m_has_payloads || lengths[c] != 0;
      if (lengths[c] == 0) m_direct_classes = uint16_t(c + 1);

      for (uint64_t p = 0; p < take; ++p) {
        d.code_of[d.value_of[next + p]] = (p << max_width) | c;
        if (lengths[c] != 0) payload_positions += sorted[next + p];
      }

      next += take;
    }

    m_eager_prefetch = size != 0 && payload_positions * eager_prefetch_divisor >= size;

    // the tables that unpack the class codes (see split_text<uint8_t>::init_unpack_tables)
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
    m_window_codes = uint8_t(std::min<uint64_t>(window_bits / m_width, batch_symbols - 1));
    m_window_codes_mask = (uint64_t{1} << (m_window_codes * m_width)) - 1;

    // the masks of the lowest and of the highest bit of each w-bit field of a 64-bit word
    for (uint64_t field = 0; field * m_width + m_width <= 64; ++field) {
      m_field_one |= uint64_t{1} << (field * m_width);
      m_field_high |= uint64_t{1} << (field * m_width + m_width - 1);
    }

    // m_direct_classes (at most 2^w - 1), split into its highest bit and its lower w - 1 bits in each field
    m_field_low = m_field_high - m_field_one;
    const uint64_t direct = std::min<uint64_t>(m_direct_classes, classes - 1);
    m_direct_high = ((direct >> (m_width - 1)) & 1) != 0;
    m_direct_low = m_field_one * (direct & ((uint64_t{1} << (m_width - 1)) - 1));
    packed_text<>::histogram_t class_histogram{};
    for (uint64_t c = 0; c < classes; ++c) class_histogram[c] = 1;
    m_class_codes = packed_text<>(class_histogram, size);

    if (m_has_payloads) {
      d.super_offsets.assign(size / super_symbols + 2, 0);
      d.block_offsets.assign(offset_bytes * (size / block_symbols + 2) + 8, 0);
    }

    d.payload.assign(padding_bytes, 0);
    cache_pointers();
  }

  /**
   * @brief returns the histogram of an integer text
   * @param data the text
   * @param size its length n
   * @param sigma the values are in [0..sigma-1]
   * @param threads number of threads
   */
  static histogram_t count_symbols(const uint32_t* data, uint64_t size, uint64_t sigma,
                                   int threads = omp_get_max_threads()) {
    histogram_t histogram(sigma, 0);

    if (sigma * uint64_t(std::max(threads, 1)) <= max_partial_histogram_entries) {
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

  /**
   * @brief stores the part T[at..at+length) of the text; parts can be packed by several threads at once
   * @param data the part T[at..at+length)
   * @param length its length
   * @param at its starting position in the text, a multiple of chunk_symbols
   * @param threads number of threads
   */
  void pack(const uint32_t* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) {
    const uint64_t* code_of = m_data->code_of.data();
    m_class_codes.pack_symbols(length, at, [code_of, data](uint64_t i) { return code_of[data[i]] & code_class_mask; }, threads);
    if (!m_has_payloads || length == 0) return;
    const uint8_t* payload_len_of_class = m_data->payload_len.data();
    const uint64_t blocks = (length + block_symbols - 1) / block_symbols;
    const uint64_t parts = std::min<uint64_t>(blocks, std::max(threads, 1));

    // each thread writes the payloads of a range of whole blocks into a piece; finish() concatenates the pieces
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
        for (uint64_t i = b * block_symbols; i < block_end; ++i) bits += payload_len_of_class[code_of[data[i]] & code_class_mask];
        piece.block_bits[b - first_block] = uint16_t(bits);
        total += bits;
      }

      piece.bits = total;
      resize_no_init(piece.words, (total + 63) / 64 + 1);
      uint64_t word = 0;
      uint64_t acc = 0; // the payload bits that do not fill a word yet
      uint64_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint64_t code = code_of[data[i]];
        const uint64_t value = code >> max_width;
        const uint64_t len = payload_len_of_class[code & code_class_mask];
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

  /**
   * @brief builds the payload stream and the directory from the pieces written by pack() and frees code_of; called
   * once after all parts have been packed
   * @param threads number of threads
   */
  void finish(int threads = omp_get_max_threads()) {
    data_t& d = *m_data;

    d.code_of = no_init_vector<uint64_t>();

    if (!m_has_payloads) {
      d.pieces = decltype(d.pieces)();
      cache_pointers();
      return;
    }

    std::sort(d.pieces.begin(), d.pieces.end(), [](const piece_t& a, const piece_t& b) { return a.at < b.at; });
    const uint64_t pieces = d.pieces.size();
    std::vector<uint64_t> offset(pieces + 1, 0); // offset[p] = payload offset of the first position of piece p
    for (uint64_t p = 0; p < pieces; ++p) offset[p + 1] = offset[p] + d.pieces[p].bits;
    const uint64_t total = offset[pieces];
    m_payload_bits = total;
    m_avg_payload_bits_x256 = uint32_t(m_size == 0 ? 0 : (total * 256 + m_size / 2) / m_size);
    resize_no_init(d.payload, (total + 63) / 64 * 8 + padding_bytes);
    advise_huge_pages(d.payload.data(), d.payload.size());
    parallel_memset(d.payload.data(), 0, d.payload.size(), threads);

    // super block offsets
#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        if (block % blocks_per_super == 0) d.super_offsets[block / blocks_per_super] = cum;
        cum += piece.block_bits[k];
      }
    }

    const uint64_t end_block = (m_size + block_symbols - 1) / block_symbols;
    if (end_block % blocks_per_super == 0) d.super_offsets[end_block / blocks_per_super] = total;

    // block offsets, relative to the offsets of their super blocks
#pragma omp parallel for num_threads(threads)
    for (uint64_t p = 0; p < pieces; ++p) {
      const piece_t& piece = d.pieces[p];
      uint64_t cum = offset[p];
      uint64_t block = piece.at / block_symbols;

      for (uint64_t k = 0; k < piece.block_bits.size(); ++k, ++block) {
        write_offset(d.block_offsets.data(), block, cum - d.super_offsets[block / blocks_per_super]);
        cum += piece.block_bits[k];
      }
    }

    write_offset(d.block_offsets.data(), end_block, total - d.super_offsets[end_block / blocks_per_super]);
    uint64_t* out = reinterpret_cast<uint64_t*>(d.payload.data());

    // copies the bits of each piece to its offset; the first and the last word of a piece can be shared with the
    // neighbouring pieces and are combined with an atomic or
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
  // number w of bits per class code
  uint8_t width() const { return m_width; }
  uint64_t penalty() const { return m_penalty; }
  bool has_payloads() const { return m_has_payloads; }
  // total number of payload bits
  uint64_t payload_bits() const { return m_payload_bits; }
  // [0..2^w-1] payload length of each class
  const uint8_t* class_lengths() const { return m_payload_len; }

  uint64_t size_in_bytes() const {
    uint64_t bytes = sizeof(*this) + m_class_codes.size_in_bytes();
    if (m_data) {
      bytes += sizeof(data_t) + m_data->payload.size() + m_data->super_offsets.size() * 8 + m_data->block_offsets.size() +
               m_data->value_of.size() * sizeof(uint32_t) + m_data->code_of.size() * sizeof(uint64_t);
    }
    return bytes;
  }

  /**
   * @brief returns size_in_bytes() of the split text of a text with the given histogram, without building it
   * @param histogram [0..sigma-1] number of occurrences of each value
   * @param size length n of the text
   * @param penalty cost in bits added for each position with a payload when choosing the code
   */
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

  // returns the symbol (the frequency rank of the value) at position i
  uint32_t operator[](uint64_t i) const {
    const uint8_t c = m_class_codes[i];
    if (m_payload_len[c] == 0) return m_first_symbol[c];
    return payload_symbol(i, c);
  }

  // returns the value at position i
  char_type char_at(uint64_t i) const { return m_value_of[(*this)[i]]; }
  // returns whether the symbol at i is smaller than the symbol at j
  bool less_char(uint64_t i, uint64_t j) const { return (*this)[i] < (*this)[j]; }
  // returns the value of a symbol
  uint32_t to_char(uint32_t symbol) const { return m_value_of[symbol]; }
  // returns a cursor whose next() returns the symbols at i, i + 1, ...
  cursor cursor_at(uint64_t i) const { return cursor(this, i); }

  /**
   * @brief writes the symbols at the positions [i, i + len) to out
   * @param i [0..n-1] position
   * @param len number of symbols
   * @param out array of at least len symbols
   */
  void extract(uint64_t i, uint64_t len, uint32_t* __restrict out) const {
    if (len == 0) return;
    if (m_has_payloads) prefetch_payload(i);
    uint64_t bit = m_has_payloads ? payload_bit(i) : 0;
    alignas(64) uint32_t buffer[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
    uint64_t skip = i - batch;
    uint64_t k = 0;

    while (k < len) {
      const uint64_t take = std::min(len - k, batch_symbols - skip);
      decode_batch(batch, skip, skip + take, bit, buffer);
      std::memcpy(out + k, buffer + skip, take * sizeof(uint32_t));
      k += take;
      batch += batch_symbols;
      skip = 0;
    }
  }

  /**
   * @brief returns the length of the longest common prefix of T[i..n) and T[j..n), at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_class_codes.lce(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t limit = std::min(max, m_size - std::max(i, j));
    if (limit == 0) [[unlikely]] return 0;

    // compare the class codes of the first m_window_codes positions
    const uint64_t wi = class_window(i);
    const uint64_t wj = class_window(j);
    const uint64_t diff = wi ^ wj;
    if ((diff & m_class_mask) != 0) return 0;

    if (m_eager_prefetch) {
      prefetch_predicted(i, diff == 0 ? prefetched_lines : 1);
      prefetch_predicted(j, diff == 0 ? prefetched_lines : 1);
    }

    // the class codes differ within the window: compare the payloads of the positions before the difference
    if (diff != 0 || limit <= m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_bit_to_symbol[std::countr_zero(diff)] : m_window_codes);
      if (!window_has_payload(wi, len)) return len;
      return payload_lce(i, j, len);
    }

    if (!m_eager_prefetch) {
      prefetch_block_offset(i);
      prefetch_block_offset(j);
    }

    // the same for the next m_window_codes positions
    const uint64_t vi = class_window(i + m_window_codes);
    const uint64_t vj = class_window(j + m_window_codes);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, m_window_codes + (diff2 != 0 ? m_bit_to_symbol[std::countr_zero(diff2)] : m_window_codes));
      if (!window_has_payload(wi, m_window_codes) && !window_has_payload(vi, len - m_window_codes)) return len;
      return payload_lce(i, j, len);
    }

    return long_lce(i, j, limit);
  }

  /**
   * @brief returns the length of the longest common suffix of T[0..i] and T[0..j], at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce_left(uint64_t i, uint64_t j, uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (!m_has_payloads) return m_class_codes.lce_left(i, j, max);
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min(max, std::min(i, j) + 1);
    if (limit == 0) [[unlikely]] return 0;
    if (std::min(i, j) + 1 < 2 * m_window_codes) [[unlikely]] return lce_left_near_start(i, j, limit);

    // as lce(), with windows that end at i and j
    const uint64_t wi = class_window(i + 1 - m_window_codes);
    const uint64_t wj = class_window(j + 1 - m_window_codes);
    const uint64_t diff = wi ^ wj;
    if ((diff >> ((m_window_codes - 1) * m_width)) != 0) return 0;

    if (m_eager_prefetch) {
      prefetch_predicted(i, diff == 0 ? -prefetched_lines : -1);
      prefetch_predicted(j, diff == 0 ? -prefetched_lines : -1);
    }

    if (diff != 0 || limit <= m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, diff != 0 ? m_window_codes - 1 - m_bit_to_symbol[63 - std::countl_zero(diff)] : m_window_codes);
      if (!window_has_payload(wi >> ((m_window_codes - len) * m_width), len)) return len;
      return payload_lce_left(i, j, len);
    }

    if (!m_eager_prefetch) {
      prefetch_block_offset(i);
      prefetch_block_offset(j);
    }

    const uint64_t vi = class_window(i + 1 - 2 * m_window_codes);
    const uint64_t vj = class_window(j + 1 - 2 * m_window_codes);
    const uint64_t diff2 = vi ^ vj;

    if (diff2 != 0 || limit <= 2 * m_window_codes) {
      const uint64_t len = std::min<uint64_t>(limit, m_window_codes + (diff2 != 0 ? m_window_codes - 1 - m_bit_to_symbol[63 - std::countl_zero(diff2)] : m_window_codes));
      const uint64_t rest = len - m_window_codes;
      if (!window_has_payload(wi, m_window_codes) && !window_has_payload(vi >> ((m_window_codes - rest) * m_width), rest)) return len;
      return payload_lce_left(i, j, len);
    }

    return long_lce_left(i, j, limit);
  }

  // returns whether T[i..i+len) = T[j..j+len)
  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  // returns a hash value of T[i..i+len), computed from the class codes and the payload bits of the range
  uint64_t hash(uint64_t i, uint64_t len) const {
    const uint64_t h = m_class_codes.hash(i, len);
    if (!m_has_payloads || len == 0) return h;
    return payload_hash(i, len, h);
  }

 private:
  // the payloads of a range of whole blocks, written by one thread in pack()
  struct piece_t {
    uint64_t at = 0; // first position of the range
    uint64_t bits = 0; // number of payload bits
    no_init_vector<uint64_t> words; // the payload bits
    std::vector<uint16_t> block_bits; // block_bits[k] = number of payload bits of the k-th block of the range
  };

  // mask of the class in an entry of code_of
  static constexpr uint64_t code_class_mask = (uint64_t{1} << max_width) - 1;

  // the arrays and tables of the text (shared by the copies of a text)
  struct data_t {
    no_init_vector<uint8_t> payload; // the payload stream, followed by padding_bytes zero bytes
    std::vector<uint64_t> super_offsets; // [0..n/super_symbols+1] payload offset of the first position of each super block
    std::vector<uint8_t> block_offsets; // [0..n/block_symbols+1] the same for each block, relative to its super block
    no_init_vector<uint32_t> value_of; // [0..sigma-1] value of symbol s
    no_init_vector<uint64_t> code_of; // [0..sigma-1] (p << max_width) | c for the value v with index p in class c; until finish()
    std::vector<piece_t> pieces; // the pieces written by pack() until finish()
    std::mutex mutex; // guards pieces
    alignas(64) std::array<uint8_t, 256> payload_len{}; // [0..2^w-1] payload length of class c
    alignas(64) std::array<uint64_t, 256> payload_mask{}; // [0..2^w-1] 2^payload_len[c] - 1
    alignas(64) std::array<uint32_t, 256> first_symbol{}; // [0..2^w-1] smallest symbol of class c
    alignas(64) std::array<uint8_t, 64> spread{}; // byte indices that unpack the class codes
    alignas(64) std::array<uint8_t, 64> shifts{}; // bit shifts or multipliers that unpack the class codes
    alignas(64) std::array<uint8_t, 64> bit_to_symbol{}; // [0..63] index of the class code that contains bit k of a window
  };

  /**
   * @brief the numbers p of most frequent symbols after which a class can end in choose_code(): every p in
   * [0..exact], then every multiple of step after exact (step is the smallest power of two with at most
   * coarse_code_positions such positions), and sigma
   */
  struct code_grid {
    std::vector<uint64_t> pos; // the grid positions in increasing order
    std::vector<uint64_t> prefix; // prefix[k] = number of occurrences of the pos[k] most frequent symbols
    uint64_t exact = 0;
    uint64_t step = 1;
    uint8_t step_shift = 0; // log2(step)

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

    // returns the index k with pos[k] = p, or -1 if p is no grid position
    int64_t index_of(uint64_t p) const {
      if (p <= exact) return int64_t(p);
      if (p == pos.back()) return int64_t(pos.size()) - 1;
      if (((p - exact) & (step - 1)) != 0) return -1;
      const uint64_t k = exact + ((p - exact) >> step_shift);
      return k < pos.size() && pos[k] == p ? int64_t(k) : -1;
    }
  };

  // sets the pointers into the tables and arrays
  void cache_pointers() {
    data_t& d = *m_data;
    m_class_code_data = m_class_codes.packed_data();
    m_class_code_data_end = m_class_code_data + (m_size * m_width + 63) / 64 * 8;
    m_payload = d.payload.data();
    m_super_offsets = d.super_offsets.data();
    m_block_offsets = d.block_offsets.data();
    m_payload_len = d.payload_len.data();
    m_payload_mask = d.payload_mask.data();
    m_first_symbol = d.first_symbol.data();
    m_value_of = d.value_of.data();
    m_spread = d.spread.data();
    m_shifts = d.shifts.data();
    m_bit_to_symbol = d.bit_to_symbol.data();
  }

  // returns the number of bits of the directory of a text of length size
  static uint64_t directory_bits(uint64_t size) {
    return (size / super_symbols + 2) * 64 + (size / block_symbols + 2) * 8 * offset_bytes;
  }

  /**
   * @brief chooses the code with the smallest cost (bits of the class codes and the payloads, penalty bits per
   * position with a payload, bits of the directory) by dynamic programming over the grid positions
   *
   * f[c][k] is the smallest cost of the payloads (l + penalty bits per occurrence for a class with payload length
   * l > 0) of the pos[k] most frequent symbols with c classes, where a class with payload length l takes the next 2^l
   * symbols and ends at a grid position (the last class can take fewer). The w * n bits of the class codes do not
   * depend on the classes, so f is computed once for up to max_classes classes, and each width w <= max_split_width
   * takes the best number of classes c <= 2^w. Widths larger than max_split_width are only possible with one class
   * per symbol.
   * @param sorted the frequencies of the symbols in decreasing order
   * @param size length n of the text
   * @param penalty cost in bits added for each position with a payload
   * @param lengths receives the payload lengths of the classes, increasing
   * @return the class width w
   */
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
    std::vector<uint8_t> from_len((max_classes + 1) * num_pos, 0); // payload length of the last class of f[c][k]
    std::vector<uint32_t> from_pos((max_classes + 1) * num_pos, 0); // grid index before the last class of f[c][k]
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

          // the class takes all remaining symbols and is the last one
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

    // the best width and number of classes
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
                          (f[at(c_min, last)] > 0 ? double(directory_bits(size)) : 0.0);

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

  // writes the relative offset of a block (offset_bytes bytes, little endian)
  static void write_offset(uint8_t* block_offsets, uint64_t block, uint64_t value) {
    for (uint64_t k = 0; k < offset_bytes; ++k) block_offsets[offset_bytes * block + k] = uint8_t(value >> (8 * k));
  }

  // returns the payload offset of the first position of a block
  uint64_t block_bit(uint64_t block) const {
    uint32_t value;
    std::memcpy(&value, m_block_offsets + offset_bytes * block, sizeof(value));
    return m_super_offsets[block / blocks_per_super] + (value & ((uint32_t{1} << (8 * offset_bytes)) - 1));
  }

  // returns the payload offset of position i: the offset of its block plus the payload lengths of the positions of
  // the block before i
  uint64_t payload_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + prefix_bits(block * block_symbols, i - block * block_symbols);
  }

  // returns an estimate of the payload offset of position i from the offset of its block and the average number of
  // payload bits per position
  uint64_t predicted_bit(uint64_t i) const {
    const uint64_t block = i / block_symbols;
    return block_bit(block) + (((i - block * block_symbols) * m_avg_payload_bits_x256) >> 8);
  }

  void prefetch_block_offset(uint64_t i) const { __builtin_prefetch(m_block_offsets + offset_bytes * (i / block_symbols)); }

  // prefetches the cache line of the estimated payload offset of position i
  void prefetch_payload(uint64_t i) const { __builtin_prefetch(m_payload + (predicted_bit(i) >> 3)); }

  // prefetches the cache lines of the payload bits [bit, bit + bits)
  void prefetch_range(uint64_t bit, uint64_t bits) const {
    const uint8_t* end = m_payload + ((bit + bits) >> 3);
    for (const uint8_t* p = m_payload + (bit >> 3); p <= end; p += 64) __builtin_prefetch(p);
  }

  // prefetches lines cache lines from the estimated payload offset of position i on, backwards if lines < 0
  void prefetch_predicted(uint64_t i, int64_t lines) const {
    const uint64_t byte = predicted_bit(i) >> 3;

    if (lines > 0) {
      for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
    } else {
      for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
    }
  }

  // returns the symbol at position i of class c > 0
  uint32_t payload_symbol(uint64_t i, uint8_t c) const {
    prefetch_payload(i);
    const uint64_t bit = payload_bit(i);
    const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
    return uint32_t(m_first_symbol[c] + payload);
  }

  // returns the class codes of the positions [i, i + m_window_codes), position i in the lowest bits
  uint64_t class_window(uint64_t i) const {
    const uint64_t bit = i * m_width;
    return (load_u64(m_class_code_data + (bit >> 3)) >> (bit & 7)) & m_window_codes_mask;
  }

  // returns whether one of the lowest len class codes in window is at least m_direct_classes, i.e., has a payload;
  // in t, the highest bit of a field is set iff the lower w - 1 bits of its class code are at least those of
  // m_direct_classes
  bool window_has_payload(uint64_t window, uint64_t len) const {
    const uint64_t x = window & ((uint64_t{1} << (len * m_width)) - 1);
    const uint64_t t = ((x & m_field_low) | m_field_high) - m_direct_low;
    const uint64_t r = (m_direct_high ? (x & t) : (x | t)) & m_field_high;
    return r != 0;
  }

  // returns the lce of T[i..] and T[j..], given that their first len class codes are equal and that it is at most len
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t pi = payload_bit(i);
    const uint64_t bits = len <= max_summed_symbols ? range_bits(i, len) : payload_bit(i + len) - pi;
    if (bits == 0) return len;
    const uint64_t pj = payload_bit(j);
    prefetch_range(pi, bits);
    prefetch_range(pj, bits);
    const uint64_t common = common_prefix(pi, pj, bits);
    if (common == bits) return len;
    return select_forward(i, common);
  }

  // returns the lce_left of T[..i] and T[..j], given that their last len class codes are equal and that it is at most
  // len
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const {
    const uint64_t first = i + 1 - len;
    const uint64_t ei = payload_bit(i + 1);
    const uint64_t bits = len <= max_summed_symbols ? range_bits(first, len) : ei - payload_bit(first);
    if (bits == 0) return len;
    const uint64_t ej = payload_bit(j + 1);
    prefetch_range(ei - bits, bits);
    prefetch_range(ej - bits, bits);
    const uint64_t common = common_suffix(ei, ej, bits);
    if (common == bits) return len;
    return select_backward(i, common);
  }

  // returns lce(i, j, limit), given that the first 2 * m_window_codes class codes are equal: compares the class codes
  // and then the payloads block_symbols positions at a time
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
      prefetch_predicted(i, prefetched_lines);
      prefetch_predicted(j, prefetched_lines);
    }

    uint64_t pi = payload_bit(i);
    uint64_t pj = payload_bit(j);
    uint64_t k = 0;

    while (k < limit) {
      const uint64_t step = std::min(limit - k, block_symbols);
      const uint64_t len = m_class_codes.lce(i + k, j + k, step);
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

  // returns lce_left(i, j, limit) block_symbols positions at a time
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const {
    if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
      prefetch_predicted(i, -prefetched_lines);
      prefetch_predicted(j, -prefetched_lines);
    }

    uint64_t ei = payload_bit(i + 1);
    uint64_t ej = payload_bit(j + 1);
    uint64_t k = 0;

    while (k < limit) {
      const uint64_t step = std::min(limit - k, block_symbols);
      const uint64_t len = m_class_codes.lce_left(i - k, j - k, step);
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

  // returns lce_left(i, j, limit) for min(i, j) < 2 * m_window_codes - 1, where the windows of lce_left() would start
  // before position 0
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t lce_left_near_start(uint64_t i, uint64_t j, uint64_t limit) const {
    return long_lce_left(i, j, limit);
  }

  // returns h mixed with the hash value of the payload bits of T[i..i+len)
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

  // returns the mask of the lanes [skip, end) of a batch
  static uint64_t lane_range(uint64_t skip, uint64_t end) {
    return (end >= batch_symbols ? ~uint64_t{0} : (uint64_t{1} << end) - 1) & (~uint64_t{0} << skip);
  }

  /**
   * @brief decodes the positions batch + [skip, end) into out[skip..end)
   * @param batch first position of the batch, a multiple of batch_symbols
   * @param position payload offset of position batch + skip; receives the payload offset of position batch + end
   */
  void decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint32_t* __restrict out) const {
    uint64_t bit = position;
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI) || defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
    if (m_width <= max_split_width) {
      alignas(64) uint8_t classes[batch_symbols];
#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
      _mm512_store_si512(classes, unpack(batch));
#else
      unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
#endif

      for (uint64_t x = skip; x < end; ++x) {
        const uint8_t c = classes[x];
        const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
        bit += m_payload_len[c];
        out[x] = uint32_t(m_first_symbol[c] + payload);
      }

      position = bit;
      return;
    }
#endif

    packed_text<>::cursor classes = m_class_codes.cursor_at(batch + skip);

    for (uint64_t x = skip; x < end; ++x) {
      const uint8_t c = classes.next();
      const uint64_t payload = (load_u64(m_payload + (bit >> 3)) >> (bit & 7)) & m_payload_mask[c];
      bit += m_payload_len[c];
      out[x] = uint32_t(m_first_symbol[c] + payload);
    }

    position = bit;
  }

#if defined(COMPACT_TEXTS_SPLIT_TEXT_VBMI)
  // returns the batch_symbols class codes stored from byte p on, one byte per class code
  __m512i unpack_at(const uint8_t* p) const {
    const __m512i raw = _mm512_loadu_si512(p);
    const __m512i spread = _mm512_permutexvar_epi8(_mm512_load_si512(m_spread), raw);
    const __m512i fields = _mm512_multishift_epi64_epi8(_mm512_load_si512(m_shifts), spread);
    return _mm512_and_si512(fields, _mm512_set1_epi8(char(m_class_mask)));
  }

  // returns the class codes of the batch starting at position first (a multiple of batch_symbols)
  __m512i unpack(uint64_t first) const { return unpack_at(m_class_code_data + (first * m_width) / 8); }

  // returns the number of payload bits of the positions [first, first + count), where first is the first position of
  // a block and count < block_symbols; the reads of the batches after the class codes start at m_class_code_data_end
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const __m512i zero = _mm512_setzero_si512();
    const uint8_t* p = m_class_code_data + (first * m_width) / 8;
    __m512i acc = zero;
    const __m512i table = _mm512_load_si512(m_payload_len);
    const uint64_t batch_bytes = batch_symbols * m_width / 8;

    for (uint64_t c = 0; c < batches_per_block; ++c) {
      const uint64_t take = std::min<uint64_t>(count > c * batch_symbols ? count - c * batch_symbols : 0, batch_symbols);
      const __mmask64 k = _cvtu64_mask64(take >= batch_symbols ? ~uint64_t{0} : (uint64_t{1} << take) - 1);
      const __m512i cls = unpack_at(std::min(p + c * batch_bytes, m_class_code_data_end));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, cls, table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  // returns the number of payload bits of the positions [i, i + len)
  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const __m512i zero = _mm512_setzero_si512();
    const __m512i table = _mm512_load_si512(m_payload_len);
    const uint64_t end = i + len;
    __m512i acc = zero;

    for (uint64_t batch = i / batch_symbols * batch_symbols; batch < end; batch += batch_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, batch_symbols);
      const __mmask64 k = _cvtu64_mask64(lane_range(lo, hi));
      acc = _mm512_add_epi64(acc, _mm512_sad_epu8(_mm512_maskz_permutexvar_epi8(k, unpack(batch), table), zero));
    }

    return uint64_t(_mm512_reduce_add_epi64(acc));
  }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    const __mmask64 k = _cvtu64_mask64(lanes);
    const __m512i l = _mm512_maskz_permutexvar_epi8(k, unpack(batch), _mm512_load_si512(m_payload_len));
    _mm512_store_si512(lens, l);
    return uint64_t(_mm512_reduce_add_epi64(_mm512_sad_epu8(l, _mm512_setzero_si512())));
  }
#elif defined(COMPACT_TEXTS_SPLIT_TEXT_AVX2)
  // returns the 16 class codes stored from byte p on, one per 16-bit element (garbage above the lowest w bits)
  __m256i unpack_pair(const uint8_t* p) const {
    const __m128i lo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p));
    const __m128i hi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(p + m_width));
    const __m256i raw = _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
    const __m256i pairs = _mm256_shuffle_epi8(raw, _mm256_load_si256(reinterpret_cast<const __m256i*>(m_spread)));
    const __m256i scale = _mm256_load_si256(reinterpret_cast<const __m256i*>(m_shifts));
    return _mm256_srli_epi16(_mm256_mullo_epi16(pairs, scale), 8);
  }

  // writes the batch_symbols class codes stored from byte p on to out, one byte per class code
  void unpack_bytes(const uint8_t* p, uint8_t* out) const {
    const __m256i mask = _mm256_set1_epi8(char(m_class_mask));
    const uint64_t step = 2 * m_width;
    const __m256i a = _mm256_packus_epi16(unpack_pair(p), unpack_pair(p + step));
    const __m256i b = _mm256_packus_epi16(unpack_pair(p + 2 * step), unpack_pair(p + 3 * step));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_and_si256(_mm256_permute4x64_epi64(a, 0xd8), mask));
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 32), _mm256_and_si256(_mm256_permute4x64_epi64(b, 0xd8), mask));
  }

  // returns table[c] for each class code c in classes; table has 2^w <= 64 entries, looked up 16 at a time
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

  // returns 0xFF in byte k if bit k of lanes is set, else 0
  static __m256i lane_bytes(uint32_t lanes) {
    const __m256i v = _mm256_shuffle_epi8(_mm256_set1_epi32(int(lanes)),
                                          _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2,
                                                           2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3));
    const __m256i bit = _mm256_set1_epi64x(int64_t(0x8040201008040201ull));
    return _mm256_cmpeq_epi8(_mm256_and_si256(v, bit), bit);
  }

  // returns the sum of the 64 bytes of a and b
  static uint64_t sum_bytes(__m256i a, __m256i b) {
    const __m256i s = _mm256_add_epi64(_mm256_sad_epu8(a, _mm256_setzero_si256()), _mm256_sad_epu8(b, _mm256_setzero_si256()));
    const __m128i t = _mm_add_epi64(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1));
    return uint64_t(_mm_cvtsi128_si64(t)) + uint64_t(_mm_extract_epi64(t, 1));
  }

  // writes the payload lengths of the 64 class codes in classes to la and lb (0 outside the mask lanes) and returns
  // their sum
  uint64_t lens_of(const uint8_t* classes, uint64_t lanes, __m256i& la, __m256i& lb) const {
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes));
    const __m256i b = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(classes + 32));
    la = _mm256_and_si256(lookup(m_payload_len, a), lane_bytes(uint32_t(lanes)));
    lb = _mm256_and_si256(lookup(m_payload_len, b), lane_bytes(uint32_t(lanes >> 32)));
    return sum_bytes(la, lb);
  }

  // returns the number of payload bits of the positions [first, first + count), where first is the first position of
  // a block and count < block_symbols
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    const uint64_t batch_bytes = batch_symbols * m_width / 8;
    const uint8_t* p = m_class_code_data + (first * m_width) / 8;
    alignas(64) uint8_t classes[batch_symbols];
    uint64_t bits = 0;

    for (uint64_t c = 0; c < batches_per_block && count > c * batch_symbols; ++c) {
      const uint64_t take = std::min<uint64_t>(count - c * batch_symbols, batch_symbols);
      unpack_bytes(std::min(p + c * batch_bytes, m_class_code_data_end), classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(0, take), la, lb);
    }

    return bits;
  }

  // returns the number of payload bits of the positions [i, i + len)
  uint64_t range_bits(uint64_t i, uint64_t len) const {
    const uint64_t end = i + len;
    alignas(64) uint8_t classes[batch_symbols];
    uint64_t bits = 0;

    for (uint64_t batch = i / batch_symbols * batch_symbols; batch < end; batch += batch_symbols) {
      const uint64_t lo = i > batch ? i - batch : 0;
      const uint64_t hi = std::min(end - batch, batch_symbols);
      unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
      __m256i la, lb;
      bits += lens_of(classes, lane_range(lo, hi), la, lb);
    }

    return bits;
  }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    alignas(64) uint8_t classes[batch_symbols];
    unpack_bytes(m_class_code_data + (batch * m_width) / 8, classes);
    __m256i la, lb;
    const uint64_t sum = lens_of(classes, lanes, la, lb);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens), la);
    _mm256_storeu_si256(reinterpret_cast<__m256i*>(lens + 32), lb);
    return sum;
  }
#else
  // returns the number of payload bits of the positions [first, first + count)
  uint64_t prefix_bits(uint64_t first, uint64_t count) const {
    uint64_t bits = 0;
    for (uint64_t x = 0; x < count; ++x) bits += m_payload_len[m_class_codes[first + x]];
    return bits;
  }

  uint64_t range_bits(uint64_t i, uint64_t len) const { return prefix_bits(i, len); }

  // writes the payload length of each position of a batch to lens (0 outside the mask lanes) and returns their sum
  uint64_t batch_lens(uint64_t batch, uint64_t lanes, uint8_t* lens) const {
    uint64_t sum = 0;

    for (uint64_t x = 0; x < batch_symbols; ++x) {
      lens[x] = (lanes >> x) & 1 ? m_payload_len[m_class_codes[batch + x]] : 0;
      sum += lens[x];
    }

    return sum;
  }
#endif

  // returns the number m of positions i, i + 1, ... before the position whose payload contains the payload bit d,
  // counted from the payload offset of i (positions without payload are counted)
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_forward(uint64_t i, uint64_t d) const {
    alignas(64) uint8_t lens[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
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
      m += batch_symbols - skip;
      batch += batch_symbols;
      skip = 0;
    }
  }

  // returns the number m of positions i, i - 1, ... after the position whose payload contains the s-th payload bit
  // before the end of the payload of i
  COMPACT_TEXTS_SPLIT_TEXT_NOINLINE uint64_t select_backward(uint64_t i, uint64_t s) const {
    alignas(64) uint8_t lens[batch_symbols];
    uint64_t batch = i / batch_symbols * batch_symbols;
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
      batch -= batch_symbols;
      last = batch_symbols - 1;
    }
  }

  // returns the number of equal bits at the starts of the payload bits [a, a + bits) and [b, b + bits)
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

  // returns the number of equal bits at the ends of the payload bits [a - bits, a) and [b - bits, b)
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

  // returns the 64-bit word of the payload stream starting at bit bit (at least window_bits + 1 valid bits)
  uint64_t load_bits(uint64_t bit) const { return load_u64(m_payload + (bit >> 3)) >> (bit & 7); }

  packed_text<> m_class_codes; // the class code of each position, w bits per position
  std::shared_ptr<data_t> m_data;
  // pointers into m_class_codes and m_data
  const uint8_t* m_class_code_data = nullptr;
  const uint8_t* m_class_code_data_end = nullptr; // end of the words of m_class_codes
  const uint8_t* m_payload = nullptr;
  const uint64_t* m_super_offsets = nullptr;
  const uint8_t* m_block_offsets = nullptr;
  const uint8_t* m_payload_len = nullptr;
  const uint64_t* m_payload_mask = nullptr;
  const uint32_t* m_first_symbol = nullptr;
  const uint32_t* m_value_of = nullptr;
  const uint8_t* m_spread = nullptr;
  const uint8_t* m_shifts = nullptr;
  const uint8_t* m_bit_to_symbol = nullptr;
  uint64_t m_size = 0; // n
  uint64_t m_payload_bits = 0; // total number of payload bits
  uint64_t m_penalty = payload_penalty;
  uint64_t m_sigma = 0;
  uint64_t m_window_codes_mask = 0; // mask of the lowest m_window_codes * w bits
  uint64_t m_field_one = 0; // the lowest bit of each w-bit field of a word
  uint64_t m_field_high = 0; // the highest bit of each w-bit field of a word
  uint64_t m_field_low = 0; // the lower w - 1 bits of each w-bit field of a word
  uint64_t m_direct_low = 0; // the lower w - 1 bits of m_direct_classes in each w-bit field of a word
  uint32_t m_avg_payload_bits_x256 = 0; // average number of payload bits per position, times 256
  uint16_t m_direct_classes = 0; // number of classes without payload (the classes 0, ..., m_direct_classes - 1)
  uint8_t m_width = 0; // w
  uint8_t m_class_mask = 0; // 2^w - 1
  uint8_t m_window_codes = 0; // number of class codes compared at once by lce() and lce_left()
  bool m_direct_high = false; // whether the highest bit of min(m_direct_classes, 2^w - 1) is set
  bool m_eager_prefetch = false; // whether lce() prefetches the payload stream at its first step
  bool m_has_payloads = false; // whether some class has a payload
};

}

#endif
