#pragma once
#include <cstdint>

namespace SymbolicRegression::Utils
{
	// Compression function for Merkle-Damgard construction.
	// This function is generated using the framework provided.
	constexpr uint64_t Mix64(uint64_t h) noexcept
	{
		h ^= h >> 23;
		h *= 0x2127599bf4325c37ULL;
		h ^= h >> 47;
		return h;
	}

	constexpr uint64_t Compress64(uint64_t h1, uint64_t h2) noexcept
	{
		return (h1 ^ Mix64(h2)) * 0x880355f21e6d1965ULL;
	}

	constexpr uint64_t Fasthash64(const void* buff, size_t len, uint64_t seed = 0x81b1f7682652da2d) noexcept
	{
		const uint64_t* pos = (const uint64_t*)buff;
		const uint64_t* end = pos + (len / 8);
		uint64_t h = seed ^ (len * 0x880355f21e6d1965ULL);

		while (pos != end)
		{
			h = Compress64(h, *pos++);
		}

		if (len & 7)
		{
			uint64_t v = 0;
			memcpy(&v, pos, (len & 7));
			h = Compress64(h, v);
		}	

		return Mix64(h);
	}

	constexpr uint32_t Fasthash32(const void* buf, size_t len, uint64_t seed = 0x81b1f7682652da2d) noexcept
	{
		// the following trick converts the 64-bit hashcode to Fermat
		// residue, which shall retain information from both the higher
		// and lower parts of hashcode.
		uint64_t h = Fasthash64(buf, len, seed);
		return (uint32_t)(h - (h >> 32));
	}
}
