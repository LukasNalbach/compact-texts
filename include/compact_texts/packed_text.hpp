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

#ifndef COMPACT_TEXTS_PACKED_TEXT_HPP
#define COMPACT_TEXTS_PACKED_TEXT_HPP

#include <omp.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <istream>
#include <limits>
#include <memory>
#include <ostream>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include <misc/memory.hpp>
#include <misc/serialize.hpp>

#include <compact_texts/common.hpp>

namespace compact_texts {

/**
 * @brief a text with w = ceil(log2 sigma) bits per symbol (at least 1)
 *
 * Symbol i is stored at the bits [i * w, (i + 1) * w) of an array of 64-bit words, followed by padding_bytes zero
 * bytes. A byte text maps the occurring characters to the symbols 0, ..., sigma - 1 in the order of the characters.
 * lce() and lce_left() compare windows of whole symbols with one 64-bit read each, or two 64-bit reads (128 bits)
 * per step if the symbols are not bytes.
 * @tparam t_symbol uint8_t (byte text) or uint32_t (integer text)
 */
template <typename t_symbol = uint8_t>
class packed_text {
  static_assert(std::is_same_v<t_symbol, uint8_t> || std::is_same_v<t_symbol, uint32_t>);

 public:
  using window_t = uint64_t;
  using symbol_type = t_symbol;
  static constexpr bool is_byte_text = sizeof(t_symbol) == 1;
  using char_type = std::conditional_t<is_byte_text, char, t_symbol>;
  using histogram_t = std::conditional_t<is_byte_text, std::array<uint64_t, 256>, std::vector<uint64_t>>;
  using mask_type = std::conditional_t<is_byte_text, uint8_t, uint64_t>;

  // number of zero bytes after the packed words
  static constexpr uint64_t padding_bytes = 64;
  // the parts passed to pack() must start at multiples of chunk_symbols, i.e., at word boundaries
  static constexpr uint64_t chunk_symbols = 8 * sizeof(uint64_t);
  // maximum number of bits per symbol
  static constexpr uint64_t max_width = 8 * sizeof(t_symbol);

  // decodes the symbols at i, i + 1, ... from one 64-bit read per window of symbols
  class cursor {
   public:
    cursor() = default;
    cursor(const uint8_t* bytes, uint64_t bit, uint8_t width, mask_type mask)
        : m_bytes(bytes), m_bit(bit), m_width(width), m_mask(mask) {}

    symbol_type next() {
      if (m_buffered == 0) [[unlikely]] refill();
      const symbol_type symbol = symbol_type(m_window) & m_mask;
      m_window >>= m_width;
      --m_buffered;
      return symbol;
    }

   private:
    // reads the whole symbols in the 64-bit word starting at bit m_bit
    void refill() {
      m_window = load_u64(m_bytes + (m_bit >> 3)) >> (m_bit & 7);
      m_buffered = uint8_t((64 - (m_bit & 7)) / m_width);
      m_bit += uint64_t(m_buffered) * m_width;
    }

    const uint8_t* m_bytes = nullptr;
    uint64_t m_bit = 0; // first bit after the symbols in m_window
    uint64_t m_window = 0; // the next m_buffered symbols in its lowest bits
    uint8_t m_buffered = 0;
    uint8_t m_width = 0;
    mask_type m_mask = 0;
  };

  packed_text() = default;

  /**
   * @brief builds the packed text of a byte text
   * @param data the text
   * @param size its length n
   * @param threads number of threads
   */
  packed_text(const char* data, uint64_t size, int threads = omp_get_max_threads())
    requires(is_byte_text)
      : packed_text(count_chars(data, size, threads), size) {
    pack(data, size, 0, threads);
  }

  /**
   * @brief builds the packed text of an integer text
   * @param data the text
   * @param size its length n
   * @param sigma the values are in [0..sigma-1]
   * @param threads number of threads
   */
  packed_text(const symbol_type* data, uint64_t size, uint64_t sigma, int threads = omp_get_max_threads())
    requires(!is_byte_text)
      : packed_text(sigma, size) {
    pack(data, size, 0, threads);
  }

  // allocates an integer text of length size with values in [0..sigma-1]; the text is then stored with pack()
  packed_text(uint64_t sigma, uint64_t size)
    requires(!is_byte_text)
      : m_size(size) {
    m_sigma = sigma;
    init();
  }

  /**
   * @brief allocates a text with the given histogram; the text is then stored with pack()
   * @param histogram number of occurrences of each character (byte text) or value (integer text)
   * @param size length n of the text
   */
  packed_text(const histogram_t& histogram, uint64_t size) : m_size(size) {
    if constexpr (is_byte_text) {
      m_sigma = 0;

      for (uint16_t c = 0; c < 256; ++c) {
        if (histogram[c] == 0) continue;
        m_to_symbol[c] = uint8_t(m_sigma);
        m_to_char[m_sigma] = uint8_t(c);
        ++m_sigma;
      }
    } else {
      m_sigma = histogram.size();
    }

    init();
  }

  // returns the histogram of a byte text of length size
  static histogram_t count_chars(const char* data, uint64_t size,
                                 int threads = omp_get_max_threads())
    requires(is_byte_text)
  {
    histogram_t histogram{};
    add_char_counts(histogram, data, size, threads);
    return histogram;
  }

  // adds the numbers of occurrences of the characters of data[0..size) to histogram
  static void add_char_counts(histogram_t& histogram, const char* data, uint64_t size,
                              int threads = omp_get_max_threads())
    requires(is_byte_text)
  {
    std::vector<histogram_t> partial(threads, histogram_t{});

#pragma omp parallel num_threads(threads)
    {
      histogram_t& local = partial[omp_get_thread_num()];
#pragma omp for
      for (uint64_t i = 0; i < size; ++i) ++local[uint8_t(data[i])];
    }

    for (const histogram_t& local : partial) {
      for (uint16_t c = 0; c < 256; ++c) histogram[c] += local[c];
    }
  }

  /**
   * @brief stores the part T[at..at+length) of a byte text; parts can be packed by several threads at once
   * @param data the part T[at..at+length)
   * @param length its length
   * @param at its starting position in the text, a multiple of chunk_symbols
   * @param threads number of threads
   */
  void pack(const char* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads())
    requires(is_byte_text)
  {
    pack_symbols(length, at, [this, data](uint64_t i) { return uint64_t(m_to_symbol[uint8_t(data[i])]); }, threads);
  }

  // stores the part T[at..at+length) of an integer text (see the byte text version)
  void pack(const symbol_type* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads())
    requires(!is_byte_text)
  {
    pack_symbols(length, at, [data](uint64_t i) { return uint64_t(data[i]); }, threads);
  }

  /**
   * @brief stores the symbols symbol_at(0), ..., symbol_at(length - 1) at the positions [at, at + length)
   * @param at a multiple of chunk_symbols
   */
  template <typename symbol_at_t>
  void pack_symbols(uint64_t length, uint64_t at, symbol_at_t symbol_at, int threads = omp_get_max_threads()) {
    uint8_t* out = m_storage->data();
    const uint64_t chunks = (length + chunk_symbols - 1) / chunk_symbols;

    // each chunk of chunk_symbols symbols fills whole words, so the threads write disjoint words
#pragma omp parallel for num_threads(threads)
    for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
      const uint64_t beg = chunk * chunk_symbols;
      const uint64_t end = std::min(beg + chunk_symbols, length);
      uint64_t word = ((at + beg) * m_width) >> 6;
      uint64_t acc = 0; // the bits that do not fill a word yet
      uint32_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint64_t symbol = symbol_at(i);
        acc |= symbol << acc_bits;
        acc_bits += m_width;

        if (acc_bits >= 64) {
          std::memcpy(out + 8 * word++, &acc, 8);
          acc_bits -= 64;
          acc = acc_bits == 0 ? 0 : symbol >> (m_width - acc_bits);
        }
      }

      if (acc_bits != 0) std::memcpy(out + 8 * word, &acc, 8);
    }
  }

  uint64_t size() const { return m_size; }
  uint64_t sigma() const { return m_sigma; }
  // number w of bits per symbol
  uint8_t width() const { return m_width; }
  uint64_t size_in_bytes() const { return sizeof(*this) + (m_storage ? m_storage->size() : 0); }
  // returns the packed words (symbol i at bit i * w)
  const uint8_t* packed_data() const { return m_bytes; }

  // writes the text as it is in memory: n, sigma, the characters of the symbols (byte texts) and the packed symbols
  void serialize(std::ostream& out) const {
    const uint64_t sigma = m_sigma;
    out.write(reinterpret_cast<const char*>(&m_size), sizeof(m_size));
    out.write(reinterpret_cast<const char*>(&sigma), sizeof(sigma));
    if constexpr (is_byte_text) out.write(reinterpret_cast<const char*>(m_to_char.data()), std::streamsize(sigma));
    write_in_chunks(out, reinterpret_cast<const char*>(m_bytes), words() * 8);
  }

  // reads a text written by serialize()
  void load(std::istream& in) {
    uint64_t sigma = 0;
    in.read(reinterpret_cast<char*>(&m_size), sizeof(m_size));
    in.read(reinterpret_cast<char*>(&sigma), sizeof(sigma));
    m_sigma = decltype(m_sigma)(sigma);

    if constexpr (is_byte_text) {
      m_to_symbol = {};
      m_to_char = {};
      in.read(reinterpret_cast<char*>(m_to_char.data()), std::streamsize(std::min<uint64_t>(sigma, m_to_char.size())));
      for (uint64_t s = 0; s < m_sigma; ++s) m_to_symbol[m_to_char[s]] = uint8_t(s);
    }

    init();
    read_in_chunks(in, reinterpret_cast<char*>(m_storage->data()), words() * 8);
  }

  // returns the symbol at position i
  symbol_type operator[](uint64_t i) const {
    if constexpr (is_byte_text) {
      if (m_width == 8) [[likely]] return m_bytes[i];
      const uint64_t bit = i * m_width;
      return uint8_t(load_u64(m_bytes + (bit >> 3)) >> (bit & 7)) & m_mask;
    } else {
      const uint64_t bit = i * m_width;
      return symbol_type((load_u64(m_bytes + (bit >> 3)) >> (bit & 7)) & m_mask);
    }
  }

  // returns the character (byte text) or the value (integer text) at position i
  char_type char_at(uint64_t i) const {
    if constexpr (is_byte_text) return char_type(m_to_char[(*this)[i]]);
    else return (*this)[i];
  }

  // returns whether T[i] < T[j]
  bool less_char(uint64_t i, uint64_t j) const { return (*this)[i] < (*this)[j]; }

  // returns the character of a symbol
  symbol_type to_char(symbol_type symbol) const {
    if constexpr (is_byte_text) return m_to_char[symbol];
    else return symbol;
  }

  // returns the symbol of a character
  symbol_type to_symbol(symbol_type c) const {
    if constexpr (is_byte_text) return m_to_symbol[c];
    else return c;
  }

  // returns a cursor whose next() returns T[i], T[i + 1], ...
  cursor cursor_at(uint64_t i) const { return cursor(m_bytes, i * m_width, m_width, m_mask); }

  /**
   * @brief returns the length of the longest common prefix of T[i..n) and T[j..n), at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce(uint64_t i, uint64_t j,
                uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t l = std::min(i, j);
    const uint64_t r = std::max(i, j);
    const uint64_t limit = std::min(max, m_size - r);
    const uint64_t narrow = m_window_symbols;
    uint64_t k = 0;

    if (narrow > limit) [[unlikely]] {
      while (k < limit && (*this)[l + k] == (*this)[r + k]) ++k;
      return k;
    }

    window_t diff = window(l) ^ window(r);
    if (diff != 0) return m_bit_to_symbol[lowest_bit(diff)];
    k = narrow;

    // steps of two 64-bit reads, if at least three of them fit; the bit offsets of l + k and r + k in their bytes,
    // sl and sr, stay the same in all steps, since a step covers a whole number of bytes
    if (wide_usable() && k + 3 * m_wide_min <= limit) {
      const uint64_t bl = (l + k) * m_width;
      const uint64_t br = (r + k) * m_width;
      const uint8_t sl = uint8_t(bl & 7);
      const uint8_t sr = uint8_t(br & 7);
      const uint64_t step = m_wide_symbols[std::max(sl, sr)];
      const uint64_t hi_mask = m_wide_mask[std::max(sl, sr)];
      const uint64_t step_bytes = (step * m_width) >> 3;
      const uint8_t* pl = m_bytes + (bl >> 3);
      const uint8_t* pr = m_bytes + (br >> 3);

      // compares the first 64 bits after sl and sr, then the remaining step * w - 64 bits
      while (k + step <= limit) {
        uint64_t l0, l1, r0, r1;
        std::memcpy(&l0, pl, 8);
        std::memcpy(&l1, pl + 8, 8);
        std::memcpy(&r0, pr, 8);
        std::memcpy(&r1, pr + 8, 8);
        uint64_t d = ((l0 >> sl) | (l1 << (63 - sl) << 1)) ^
                     ((r0 >> sr) | (r1 << (63 - sr) << 1));
        if (d != 0) return k + m_bit_to_symbol[lowest_bit(d)];
        d = ((l1 >> sl) ^ (r1 >> sr)) & hi_mask;
        if (d != 0) return k + m_bit_to_symbol_hi[lowest_bit(d)];
        k += step;
        pl += step_bytes;
        pr += step_bytes;
      }
    }

    while (k + narrow <= limit) {
      diff = window(l + k) ^ window(r + k);
      if (diff != 0) return k + m_bit_to_symbol[lowest_bit(diff)];
      k += narrow;
    }

    // the last window ends at limit and overlaps the compared ones
    if (k == limit) return k;
    k = limit - narrow;
    diff = window(l + k) ^ window(r + k);
    return diff == 0 ? limit : k + m_bit_to_symbol[lowest_bit(diff)];
  }

  /**
   * @brief returns the length of the longest common suffix of T[0..i] and T[0..j], at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce_left(uint64_t i, uint64_t j,
                     uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min({max, i + 1, j + 1});
    const uint64_t narrow = m_window_symbols;
    uint64_t k = 0;

    if (narrow > limit) [[unlikely]] {
      while (k < limit && (*this)[i - k] == (*this)[j - k]) ++k;
      return k;
    }

    // as lce(), with windows that end at i + 1 - k and j + 1 - k
    window_t diff = window(i + 1 - narrow) ^ window(j + 1 - narrow);
    if (diff != 0) return narrow - 1 - m_bit_to_symbol[highest_bit(diff)];
    k = narrow;

    if (wide_usable() && k + 3 * m_wide_min <= limit) {
      const uint64_t bl = (i + 1 - k) * m_width;
      const uint64_t br = (j + 1 - k) * m_width;
      const uint8_t sl = uint8_t(bl & 7);
      const uint8_t sr = uint8_t(br & 7);
      const uint64_t step = m_wide_symbols[std::max(sl, sr)];
      const uint64_t hi_mask = m_wide_mask[std::max(sl, sr)];
      const uint64_t step_bytes = (step * m_width) >> 3;
      const uint8_t* pl = m_bytes + (bl >> 3);
      const uint8_t* pr = m_bytes + (br >> 3);

      while (k + step <= limit) {
        pl -= step_bytes;
        pr -= step_bytes;
        uint64_t l0, l1, r0, r1;
        std::memcpy(&l0, pl, 8);
        std::memcpy(&l1, pl + 8, 8);
        std::memcpy(&r0, pr, 8);
        std::memcpy(&r1, pr + 8, 8);
        uint64_t d = ((l1 >> sl) ^ (r1 >> sr)) & hi_mask;
        if (d != 0) return k + step - 1 - m_bit_to_symbol_hi[highest_bit(d)];
        d = ((l0 >> sl) | (l1 << (63 - sl) << 1)) ^
            ((r0 >> sr) | (r1 << (63 - sr) << 1));
        if (d != 0) return k + step - 1 - m_bit_to_symbol[highest_bit(d)];
        k += step;
      }
    }

    while (k + narrow <= limit) {
      diff = window(i + 1 - k - narrow) ^ window(j + 1 - k - narrow);
      if (diff != 0) return k + narrow - 1 - m_bit_to_symbol[highest_bit(diff)];
      k += narrow;
    }

    if (k == limit) return k;
    diff = window(i + 1 - limit) ^ window(j + 1 - limit);
    return diff == 0 ? limit : limit - 1 - m_bit_to_symbol[highest_bit(diff)];
  }

  // returns whether T[i..i+len) = T[j..j+len)
  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  // returns a hash value of the bits of T[i..i+len)
  uint64_t hash(uint64_t i, uint64_t len) const {
    // the whole bytes of a 64-bit word read at the byte of the first bit, after shifting out the bits before it
    constexpr uint64_t chunk_bits = 8 * (sizeof(uint64_t) - 1);
    constexpr uint64_t chunk_mask = (uint64_t(1) << chunk_bits) - 1;
    const uint64_t beg = i * m_width;
    const uint64_t bits = len * m_width;
    const uint64_t full = bits / chunk_bits;
    auto chunk_at = [this](uint64_t bit) { return load_u64(m_bytes + (bit >> 3)) >> (bit & 7); };
    const uint64_t h = hash_words(full, [&](uint64_t k) { return chunk_at(beg + chunk_bits * k) & chunk_mask; });
    const uint64_t rest = bits - chunk_bits * full;
    if (rest == 0) return h;
    return hash_mix(h, chunk_at(beg + chunk_bits * full) & ((uint64_t(1) << rest) - 1));
  }

  // reverses the text
  void reverse(int threads = omp_get_max_threads()) {
    auto reversed = std::make_shared<no_init_vector<uint8_t>>();
    resize_no_init(*reversed, m_storage->size());
    std::memset(reversed->data() + words() * 8, 0, padding_bytes);
    uint8_t* out = reversed->data();
    const uint64_t chunks = (m_size + chunk_symbols - 1) / chunk_symbols;

#pragma omp parallel for num_threads(threads)
    for (uint64_t chunk = 0; chunk < chunks; ++chunk) {
      const uint64_t beg = chunk * chunk_symbols;
      const uint64_t end = std::min(beg + chunk_symbols, m_size);
      uint64_t word = (beg * m_width) >> 6;
      uint64_t acc = 0;
      uint32_t acc_bits = 0;

      for (uint64_t i = beg; i < end; ++i) {
        const uint64_t symbol = (*this)[m_size - 1 - i];
        acc |= symbol << acc_bits;
        acc_bits += m_width;

        if (acc_bits >= 64) {
          std::memcpy(out + 8 * word++, &acc, 8);
          acc_bits -= 64;
          acc = acc_bits == 0 ? 0 : symbol >> (m_width - acc_bits);
        }
      }

      if (acc_bits != 0) std::memcpy(out + 8 * word, &acc, 8);
    }

    m_storage = std::move(reversed);
    m_bytes = m_storage->data();
  }

 private:
  // computes w, the window and two-word step tables and allocates the packed words
  void init() {
    m_width = uint8_t(std::max<int>(1, std::bit_width(uint64_t(std::max<uint64_t>(m_sigma, 1) - 1))));
    m_mask = mask_type((uint64_t{1} << m_width) - 1);

    // a symbol starts at a bit offset in its byte that is a multiple of gcd(w, 8), so a 64-bit read at its byte
    // contains at least 64 - (8 - gcd(w, 8)) of its bits
    m_window_symbols = uint8_t((64 - (8 - std::gcd<int>(m_width, 8))) / m_width);
    const uint64_t window_bits = uint64_t{m_window_symbols} * m_width;
    m_window_mask = window_bits == 64 ? ~window_t(0) : (window_t(1) << window_bits) - 1;

    for (uint16_t bit = 0; bit < 64; ++bit) {
      m_bit_to_symbol[bit] = uint8_t(bit / m_width);
      m_bit_to_symbol_hi[bit] = uint8_t((64 + bit) / m_width);
    }

    // a group of 8 / gcd(w, 8) symbols covers a whole number of bytes; a two-word step covers as many groups as fit
    // into the 128 - shift bits after a bit offset shift
    const uint8_t group = uint8_t(8 / std::gcd<int>(m_width, 8));
    const uint8_t group_bits = uint8_t(group * m_width);

    for (uint8_t shift = 0; shift < 8; ++shift) {
      const uint8_t step = uint8_t(group * ((128 - shift) / group_bits));
      const uint32_t bits = uint32_t(step) * m_width - 64;
      m_wide_symbols[shift] = step;
      m_wide_mask[shift] = bits >= 64 ? ~window_t(0) : ((window_t(1) << bits) - 1);
      m_wide_min = shift == 0 ? step : std::min(m_wide_min, step);
    }

    m_storage = std::make_shared<no_init_vector<uint8_t>>();
    resize_no_init(*m_storage, words() * 8 + padding_bytes);
    advise_huge_pages(m_storage->data(), m_storage->size());
    std::memset(m_storage->data() + words() * 8, 0, padding_bytes);
    m_bytes = m_storage->data();
  }

  // returns the number of 64-bit words of the packed symbols
  uint64_t words() const { return (m_size * m_width + 63) / 64; }

  // returns whether lce() and lce_left() can compare in two-word steps
  bool wide_usable() const {
    if constexpr (is_byte_text) return m_width != 8;
    else return m_width != 8 && m_wide_min != 0;
  }

  // returns the m_window_symbols symbols starting at position i, position i in the lowest bits
  window_t window(uint64_t i) const {
    if constexpr (is_byte_text) {
      if (m_width == 8) [[likely]] return load_u64(m_bytes + i);
    }
    const uint64_t bit = i * m_width;
    return (load_u64(m_bytes + (bit >> 3)) >> (bit & 7)) & m_window_mask;
  }

  static uint8_t lowest_bit(window_t diff) { return uint8_t(std::countr_zero(diff)); }

  static uint8_t highest_bit(window_t diff) { return uint8_t(63 - std::countl_zero(diff)); }

  std::shared_ptr<no_init_vector<uint8_t>> m_storage; // the packed words, followed by padding_bytes zero bytes
  const uint8_t* m_bytes = nullptr; // m_storage->data()
  uint64_t m_size = 0; // n
  window_t m_window_mask = 0; // mask of the lowest m_window_symbols * w bits
  std::conditional_t<is_byte_text, uint16_t, uint64_t> m_sigma = 0;
  uint8_t m_width = 0; // w
  mask_type m_mask = 0; // 2^w - 1
  uint8_t m_window_symbols = 0; // number of symbols compared per 64-bit read
  uint8_t m_wide_min = 0; // minimum of m_wide_symbols
  std::array<uint8_t, 64> m_bit_to_symbol{}; // [0..63] index of the symbol that contains bit k of a window
  std::array<uint8_t, 64> m_bit_to_symbol_hi{}; // [0..63] the same for bit 64 + k of a two-word step
  std::array<uint8_t, 8> m_wide_symbols{}; // [0..7] number of symbols per two-word step at bit offset shift
  std::array<window_t, 8> m_wide_mask{}; // [0..7] mask of the bits of the second word of a step at bit offset shift
  std::array<uint8_t, 256> m_to_symbol{}; // [0..255] symbol of character c (byte text)
  std::array<uint8_t, 256> m_to_char{}; // [0..sigma-1] character of symbol s (byte text)
};

}

#endif
