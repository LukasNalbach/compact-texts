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

#ifndef COMPACT_TEXTS_DIRECT_TEXT_HPP
#define COMPACT_TEXTS_DIRECT_TEXT_HPP

#include <omp.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>

#include <misc/packed_integers.hpp>

#include <compact_texts/common.hpp>

namespace compact_texts {

template <typename t_char_type>
struct direct_symbol {
  using type = std::conditional_t<sizeof(t_char_type) == 1, uint8_t, std::make_unsigned_t<t_char_type>>;
};

template <uint8_t t_bytes>
struct direct_symbol<packed_unsigned<t_bytes>> {
  using type = uint64_t;
};

// true <=> a direct_text stores symbols of type t_char_type: integers of 1, 2, 4 or 8 bytes, or packed unsigned
// integers of 3, 5, 6 or 7 bytes (see packed_unsigned)
template <typename t_char_type>
inline constexpr bool is_direct_char_v =
    (std::is_integral_v<t_char_type> && sizeof(t_char_type) <= sizeof(uint64_t)) ||
    is_packed_unsigned_v<t_char_type>;

template <typename t_char_type = char>
class direct_text {
  static_assert(is_direct_char_v<t_char_type>);

 public:
  using char_type = t_char_type;
  static constexpr bool is_byte_text = sizeof(t_char_type) == 1;
  using symbol_type = typename direct_symbol<t_char_type>::type;
  static constexpr uint64_t symbol_bytes = sizeof(t_char_type);
  static constexpr uint64_t word_symbols = 8 / symbol_bytes;
  static constexpr bool word_aligned = 8 % symbol_bytes == 0;

  class cursor {
   public:
    cursor() = default;
    explicit cursor(const char_type* at) : m_at(at) {}
    symbol_type next() { return symbol_type(*m_at++); }

   private:
    const char_type* m_at = nullptr;
  };

  direct_text() = default;
  direct_text(char_type* data, uint64_t size) : m_data(data), m_size(size) {}

  direct_text(char_type* data, uint64_t size, uint64_t sigma)
    requires(!is_byte_text)
      : m_data(data), m_size(size), m_sigma(sigma) {}

  char_type* data() const { return m_data; }
  uint64_t size() const { return m_size; }

  uint64_t sigma() const {
    if constexpr (is_byte_text) return 256;
    else return m_sigma;
  }

  uint64_t size_in_bytes() const { return m_size * symbol_bytes; }

  symbol_type operator[](uint64_t i) const { return symbol_type(m_data[i]); }
  char_type char_at(uint64_t i) const { return m_data[i]; }
  bool less_char(uint64_t i, uint64_t j) const { return m_data[i] < m_data[j]; }
  symbol_type to_char(symbol_type symbol) const { return symbol; }
  cursor cursor_at(uint64_t i) const { return cursor(m_data + i); }

  uint64_t lce(uint64_t i, uint64_t j,
               uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (i == j) [[unlikely]] return std::min(max, m_size - i);
    const uint64_t l = std::min(i, j);
    const uint64_t r = std::max(i, j);
    const uint64_t limit = std::min(max, m_size - r);

    if constexpr (!word_aligned) {
      const uint8_t* a = bytes() + l * symbol_bytes;
      const uint8_t* b = bytes() + r * symbol_bytes;
      const uint64_t limit_bytes = limit * symbol_bytes;
      uint64_t k = 0;

      while (k + 8 <= limit_bytes) {
        const uint64_t diff = load_u64(a + k) ^ load_u64(b + k);
        if (diff != 0) return (k + std::countr_zero(diff) / 8) / symbol_bytes;
        k += 8;
      }

      k /= symbol_bytes;
      while (k < limit && (*this)[l + k] == (*this)[r + k]) ++k;
      return k;
    } else {
      const symbol_type* a = symbols() + l;
      const symbol_type* b = symbols() + r;
      uint64_t k = 0;

      while (k + word_symbols <= limit) {
        const uint64_t diff = load_u64(a + k) ^ load_u64(b + k);
        if (diff != 0) return k + std::countr_zero(diff) / (8 * symbol_bytes);
        k += word_symbols;
      }

      while (k < limit && a[k] == b[k]) ++k;
      return k;
    }
  }

  uint64_t lce_left(uint64_t i, uint64_t j,
                    uint64_t max = std::numeric_limits<uint64_t>::max()) const {
    if (i == j) [[unlikely]] return std::min(max, i + 1);
    const uint64_t limit = std::min({max, i + 1, j + 1});

    if constexpr (!word_aligned) {
      const uint8_t* a = bytes() + (i + 1) * symbol_bytes;
      const uint8_t* b = bytes() + (j + 1) * symbol_bytes;
      const uint64_t limit_bytes = limit * symbol_bytes;
      uint64_t k = 0;

      while (k + 8 <= limit_bytes) {
        const uint64_t diff = load_u64(a - k - 8) ^ load_u64(b - k - 8);
        if (diff != 0) return (k + std::countl_zero(diff) / 8) / symbol_bytes;
        k += 8;
      }

      k /= symbol_bytes;
      while (k < limit && (*this)[i - k] == (*this)[j - k]) ++k;
      return k;
    } else {
      const symbol_type* a = symbols() + i + 1;
      const symbol_type* b = symbols() + j + 1;
      uint64_t k = 0;

      while (k + word_symbols <= limit) {
        const uint64_t diff = load_u64(a - k - word_symbols) ^ load_u64(b - k - word_symbols);
        if (diff != 0) return k + std::countl_zero(diff) / (8 * symbol_bytes);
        k += word_symbols;
      }

      while (k < limit && a[-1 - int64_t(k)] == b[-1 - int64_t(k)]) ++k;
      return k;
    }
  }

  bool equal(uint64_t i, uint64_t j, uint64_t len) const {
    return std::memcmp(m_data + i, m_data + j, len * symbol_bytes) == 0;
  }

  uint64_t hash(uint64_t i, uint64_t len) const {
    const uint8_t* p = bytes() + i * symbol_bytes;
    const uint64_t len_bytes = len * symbol_bytes;
    const uint64_t full = len_bytes >> 3;
    const uint64_t h = hash_words(full, [p](uint64_t k) { return load_u64(p + 8 * k); });
    if ((len_bytes & 7) == 0) return h;
    uint64_t tail = 0;
    std::memcpy(&tail, p + 8 * full, len_bytes & 7);
    return hash_mix(h, tail);
  }

  void reverse(int threads = omp_get_max_threads()) {
    const uint64_t half = m_size / 2;
#pragma omp parallel for num_threads(threads)
    for (uint64_t k = 0; k < half; ++k) std::swap(m_data[k], m_data[m_size - 1 - k]);
  }

 private:
  const uint8_t* bytes() const { return reinterpret_cast<const uint8_t*>(m_data); }
  const symbol_type* symbols() const { return reinterpret_cast<const symbol_type*>(m_data); }

  struct no_sigma {};

  char_type* m_data = nullptr;
  uint64_t m_size = 0;
  [[no_unique_address]] std::conditional_t<is_byte_text, no_sigma, uint64_t> m_sigma{};
};

template <typename t_text>
concept text_access = requires(t_text const& text) {
  text.size();
  text[0];
  text.cursor_at(0);
  text.lce(0, 0, 0);
};

template <typename t_type>
struct text_of {
  using type = t_type;
};

template <typename t_type>
  requires(is_direct_char_v<t_type>)
struct text_of<t_type> {
  using type = direct_text<t_type>;
};

template <typename t_type>
using text_of_t = typename text_of<t_type>::type;

template <typename t_text>
inline constexpr bool is_direct_text_v = false;

template <typename t_char_type>
inline constexpr bool is_direct_text_v<direct_text<t_char_type>> = true;

}

#endif
