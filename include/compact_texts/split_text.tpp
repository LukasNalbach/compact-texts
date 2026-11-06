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

#ifndef COMPACT_TEXTS_SPLIT_TEXT_TPP
#define COMPACT_TEXTS_SPLIT_TEXT_TPP

#include <compact_texts/split_text.hpp>

namespace compact_texts {

inline uint8_t split_text<uint8_t>::cursor::next()
{
  if (m_at == batch_symbols) [[unlikely]] {
    m_batch += batch_symbols;
    m_at = 0;
    m_text->template decode_batch<false>(m_batch, 0, batch_symbols, m_bit, m_buffer);
  }

  return m_buffer[m_at++];
}

inline split_text<uint8_t>::split_text(const histogram_t& histogram, uint64_t size, uint64_t penalty)
    : m_size(size), m_penalty(penalty)
{
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

inline void split_text<uint8_t>::pack(const char* data, uint64_t length, uint64_t at, int threads)
{
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

inline void split_text<uint8_t>::finish(int threads)
{
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

inline uint64_t split_text<uint8_t>::size_in_bytes() const
{
  uint64_t bytes = sizeof(*this) + m_class_codes.size_in_bytes();
  if (m_data) {
    bytes += sizeof(data_t) + m_data->payload.size() + m_data->super_offsets.size() * 8 + m_data->block_offsets.size() +
             m_data->symbol_of.size() + m_data->char_of.size();
  }
  return bytes;
}

inline void split_text<uint8_t>::serialize(std::ostream& out) const
{
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

inline void split_text<uint8_t>::load(std::istream& in)
{
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

inline uint64_t split_text<uint8_t>::size_in_bytes_for(const histogram_t& histogram, uint64_t size, uint64_t penalty)
{
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

inline void split_text<uint8_t>::extract(uint64_t i, uint64_t len, uint8_t* __restrict out) const
{
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

inline uint64_t split_text<uint8_t>::lce(uint64_t i, uint64_t j, uint64_t max) const
{
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

inline uint64_t split_text<uint8_t>::lce_left(uint64_t i, uint64_t j, uint64_t max) const
{
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

template <typename text_t, typename tables_t, typename io_t>
inline void split_text<uint8_t>::for_each_member(text_t& t, tables_t& d, io_t io)
{
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

inline void split_text<uint8_t>::init_unpack_tables()
{
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

inline void split_text<uint8_t>::cache_pointers()
{
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

inline uint8_t split_text<uint8_t>::choose_code(const std::vector<uint64_t>& sorted, uint64_t size, uint64_t penalty,
    std::vector<uint8_t>& lengths)
{
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

inline double split_text<uint8_t>::optimal_lengths(const std::vector<uint64_t>& sorted, uint8_t width, uint64_t penalty,
    std::vector<uint8_t>& lengths)
{
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

inline void split_text<uint8_t>::prefetch_predicted(uint64_t i, int64_t lines) const
{
  const uint64_t byte = predicted_bit(i) >> 3;

  if (lines > 0) {
    for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
  } else {
    for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
  }
}

inline uint64_t split_text<uint8_t>::payload_lce(uint64_t i, uint64_t j, uint64_t len) const
{
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

inline uint64_t split_text<uint8_t>::payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const
{
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

inline uint64_t split_text<uint8_t>::long_lce(uint64_t i, uint64_t j, uint64_t limit) const
{
  if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
    prefetch_predicted(i, prefetched_lines);
    prefetch_predicted(j, prefetched_lines);
  }

  const uint64_t skip = 2 * m_window_codes;
  const uint64_t len = skip + m_class_codes.lce(i + skip, j + skip, limit - skip);
  return payload_lce(i, j, len);
}

inline uint64_t split_text<uint8_t>::long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const
{
  if (!m_eager_prefetch && m_avg_payload_bits_x256 >= min_prefetch_avg_x256) {
    prefetch_predicted(i, -prefetched_lines);
    prefetch_predicted(j, -prefetched_lines);
  }

  const uint64_t skip = 2 * m_window_codes;
  const uint64_t len = skip + m_class_codes.lce_left(i - skip, j - skip, limit - skip);
  return payload_lce_left(i, j, len);
}

inline uint64_t split_text<uint8_t>::payload_hash(uint64_t i, uint64_t len, uint64_t h) const
{
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

template <bool t_chars>
inline void split_text<uint8_t>::decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint8_t* __restrict out) const
{
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

inline uint64_t split_text<uint8_t>::select_forward(uint64_t i, uint64_t d) const
{
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

inline uint64_t split_text<uint8_t>::select_backward(uint64_t i, uint64_t s) const
{
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

inline uint64_t split_text<uint8_t>::common_prefix(uint64_t a, uint64_t b, uint64_t bits) const
{
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

inline uint64_t split_text<uint8_t>::common_suffix(uint64_t a, uint64_t b, uint64_t bits) const
{
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

inline uint32_t split_text<uint32_t>::cursor::next()
{
  if (m_at == batch_symbols) [[unlikely]] {
    m_batch += batch_symbols;
    m_at = 0;
    m_text->decode_batch(m_batch, 0, batch_symbols, m_bit, m_buffer);
  }

  return m_buffer[m_at++];
}

inline split_text<uint32_t>::split_text(const histogram_t& histogram, uint64_t size, uint64_t penalty)
    : m_size(size), m_penalty(penalty)
{
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

inline auto split_text<uint32_t>::count_symbols(const uint32_t* data, uint64_t size, uint64_t sigma, int threads) -> histogram_t
{
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

inline void split_text<uint32_t>::pack(const uint32_t* data, uint64_t length, uint64_t at, int threads)
{
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

inline void split_text<uint32_t>::finish(int threads)
{
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

inline uint64_t split_text<uint32_t>::size_in_bytes() const
{
  uint64_t bytes = sizeof(*this) + m_class_codes.size_in_bytes();
  if (m_data) {
    bytes += sizeof(data_t) + m_data->payload.size() + m_data->super_offsets.size() * 8 + m_data->block_offsets.size() +
             m_data->value_of.size() * sizeof(uint32_t) + m_data->code_of.size() * sizeof(uint64_t);
  }
  return bytes;
}

inline uint64_t split_text<uint32_t>::size_in_bytes_for(const histogram_t& histogram, uint64_t size, uint64_t penalty)
{
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

inline void split_text<uint32_t>::extract(uint64_t i, uint64_t len, uint32_t* __restrict out) const
{
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

inline uint64_t split_text<uint32_t>::lce(uint64_t i, uint64_t j, uint64_t max) const
{
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

inline uint64_t split_text<uint32_t>::lce_left(uint64_t i, uint64_t j, uint64_t max) const
{
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

inline split_text<uint32_t>::code_grid::code_grid(const std::vector<uint64_t>& sorted)
{
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

inline int64_t split_text<uint32_t>::code_grid::index_of(uint64_t p) const
{
  if (p <= exact) return int64_t(p);
  if (p == pos.back()) return int64_t(pos.size()) - 1;
  if (((p - exact) & (step - 1)) != 0) return -1;
  const uint64_t k = exact + ((p - exact) >> step_shift);
  return k < pos.size() && pos[k] == p ? int64_t(k) : -1;
}

inline void split_text<uint32_t>::cache_pointers()
{
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

inline uint8_t split_text<uint32_t>::choose_code(const std::vector<uint64_t>& sorted, uint64_t size, uint64_t penalty,
    std::vector<uint8_t>& lengths)
{
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

inline void split_text<uint32_t>::prefetch_predicted(uint64_t i, int64_t lines) const
{
  const uint64_t byte = predicted_bit(i) >> 3;

  if (lines > 0) {
    for (int64_t k = 0; k < lines; ++k) __builtin_prefetch(m_payload + byte + 64 * k);
  } else {
    for (int64_t k = 0; k < -lines && byte >= uint64_t(64 * k); ++k) __builtin_prefetch(m_payload + byte - 64 * k);
  }
}

inline uint64_t split_text<uint32_t>::payload_lce(uint64_t i, uint64_t j, uint64_t len) const
{
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

inline uint64_t split_text<uint32_t>::payload_lce_left(uint64_t i, uint64_t j, uint64_t len) const
{
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

inline uint64_t split_text<uint32_t>::long_lce(uint64_t i, uint64_t j, uint64_t limit) const
{
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

inline uint64_t split_text<uint32_t>::long_lce_left(uint64_t i, uint64_t j, uint64_t limit) const
{
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

inline uint64_t split_text<uint32_t>::payload_hash(uint64_t i, uint64_t len, uint64_t h) const
{
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

inline void split_text<uint32_t>::decode_batch(uint64_t batch, uint64_t skip, uint64_t end, uint64_t& position, uint32_t* __restrict out) const
{
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

inline uint64_t split_text<uint32_t>::select_forward(uint64_t i, uint64_t d) const
{
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

inline uint64_t split_text<uint32_t>::select_backward(uint64_t i, uint64_t s) const
{
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

inline uint64_t split_text<uint32_t>::common_prefix(uint64_t a, uint64_t b, uint64_t bits) const
{
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

inline uint64_t split_text<uint32_t>::common_suffix(uint64_t a, uint64_t b, uint64_t bits) const
{
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

} // namespace compact_texts

#endif // COMPACT_TEXTS_SPLIT_TEXT_TPP
