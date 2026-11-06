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

    symbol_type next();

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
  packed_text(const char* data, uint64_t size, int threads = omp_get_max_threads()) requires(is_byte_text)
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
  packed_text(const symbol_type* data, uint64_t size, uint64_t sigma, int threads = omp_get_max_threads()) requires(!is_byte_text)
      : packed_text(sigma, size) {
    pack(data, size, 0, threads);
  }

  // allocates an integer text of length size with values in [0..sigma-1]; the text is then stored with pack()
  packed_text(uint64_t sigma, uint64_t size) requires(!is_byte_text)
      : m_size(size) {
    m_sigma = sigma;
    init();
  }

  /**
   * @brief allocates a text with the given histogram; the text is then stored with pack()
   * @param histogram number of occurrences of each character (byte text) or value (integer text)
   * @param size length n of the text
   */
  packed_text(const histogram_t& histogram, uint64_t size);

  // returns the histogram of a byte text of length size
  static histogram_t count_chars(const char* data, uint64_t size,
                                 int threads = omp_get_max_threads()) requires(is_byte_text)
  {
    histogram_t histogram{};
    add_char_counts(histogram, data, size, threads);
    return histogram;
  }

  // adds the numbers of occurrences of the characters of data[0..size) to histogram
  static void add_char_counts(histogram_t& histogram, const char* data, uint64_t size,
                              int threads = omp_get_max_threads()) requires(is_byte_text);

  /**
   * @brief stores the part T[at..at+length) of a byte text; parts can be packed by several threads at once
   * @param data the part T[at..at+length)
   * @param length its length
   * @param at its starting position in the text, a multiple of chunk_symbols
   * @param threads number of threads
   */
  void pack(const char* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) requires(is_byte_text)
  {
    pack_symbols(length, at, [this, data](uint64_t i) { return uint64_t(m_to_symbol[uint8_t(data[i])]); }, threads);
  }

  // stores the part T[at..at+length) of an integer text (see the byte text version)
  void pack(const symbol_type* data, uint64_t length, uint64_t at, int threads = omp_get_max_threads()) requires(!is_byte_text)
  {
    pack_symbols(length, at, [data](uint64_t i) { return uint64_t(data[i]); }, threads);
  }

  /**
   * @brief stores the symbols symbol_at(0), ..., symbol_at(length - 1) at the positions [at, at + length)
   * @param at a multiple of chunk_symbols
   */
  template <typename symbol_at_t>
  void pack_symbols(uint64_t length, uint64_t at, symbol_at_t symbol_at, int threads = omp_get_max_threads());

  uint64_t size() const { return m_size; }
  uint64_t sigma() const { return m_sigma; }
  // number w of bits per symbol
  uint8_t width() const { return m_width; }
  uint64_t size_in_bytes() const { return sizeof(*this) + (m_storage ? m_storage->size() : 0); }
  // returns the packed words (symbol i at bit i * w)
  const uint8_t* packed_data() const { return m_bytes; }

  // writes the text as it is in memory: n, sigma, the characters of the symbols (byte texts) and the packed symbols
  void serialize(std::ostream& out) const;

  // reads a text written by serialize()
  void load(std::istream& in);

  // returns the symbol at position i
  symbol_type operator[](uint64_t i) const;

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
                uint64_t max = std::numeric_limits<uint64_t>::max()) const;

  /**
   * @brief returns the length of the longest common suffix of T[0..i] and T[0..j], at most max
   * @param i [0..n-1] position
   * @param j [0..n-1] position
   * @param max upper bound for the result
   */
  uint64_t lce_left(uint64_t i, uint64_t j,
                     uint64_t max = std::numeric_limits<uint64_t>::max()) const;

  // returns whether T[i..i+len) = T[j..j+len)
  bool equal(uint64_t i, uint64_t j, uint64_t len) const { return lce(i, j, len) == len; }

  // returns a hash value of the bits of T[i..i+len)
  uint64_t hash(uint64_t i, uint64_t len) const;

  // reverses the text
  void reverse(int threads = omp_get_max_threads());

 private:
  // computes w, the window and two-word step tables and allocates the packed words
  void init();

  // returns the number of 64-bit words of the packed symbols
  uint64_t words() const { return (m_size * m_width + 63) / 64; }

  // returns whether lce() and lce_left() can compare in two-word steps
  bool wide_usable() const {
    if constexpr (is_byte_text) return m_width != 8;
    else return m_width != 8 && m_wide_min != 0;
  }

  // returns the m_window_symbols symbols starting at position i, position i in the lowest bits
  window_t window(uint64_t i) const;

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

#include <compact_texts/packed_text.tpp>

#endif
