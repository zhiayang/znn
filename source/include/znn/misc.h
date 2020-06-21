// misc.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "precompile.h"

namespace znn
{
	using xarr = xt::xarray<double>;

	struct Layer
	{
		virtual ~Layer() { }

		virtual xarr forward(const xarr& in) = 0;
		virtual std::pair<xarr, xarr> backward(const xarr& last, const xarr& del) = 0;
		virtual void update_weights(const xarr& w, const xarr& del) = 0;

		Layer* input_layer = 0;
		Layer* output_layer = 0;

	protected:
		Layer(Layer* in) : input_layer(in) { }
	};










	namespace util
	{
		// perform matrix multiplication, or dot product.
		xarr dot(const xarr& a, const xarr& b)
		{
			// xtensor * is a hadamard product (ie. elementwise),
			// so we need to sum along the last axis to get a dot product. note that we
			// rely on broadcasting to do this properly.

			assert(a.dimension() <= 2 && b.dimension() <= 2);

			// we do not handle matrix multiplication (yet?)
			if(a.dimension() == 2 && b.dimension() == 2)
				assert(!"cannot do matrix multiplication!");

			if(a.dimension() == 2)
			{
				assert(b.dimension() == 1);
				assert(a.shape()[1] == b.shape()[0]);

				// this uses broadcasting. collapse along the second axis.
				return xt::eval(xt::sum(a * b, 1));
			}
			else if(b.dimension() == 2)
			{
				assert(a.dimension() == 1);
				assert(a.shape()[0] == b.shape()[0]);

				// we need to transpose, multiply with broadcast, then transpose back.
				// return xt::eval(xt::transpose(b * xt::transpose(a)));
				return xt::eval(xt::sum(xt::transpose(a * xt::transpose(b)), 0));
			}
			else
			{
				auto an = a.shape()[0];
				auto bn = b.shape()[0];

				// obviously, there's an ambiguity here; for two vectors u and v,
				// u.v is either a matrix or a scalar. since xtensor vectors are
				// 1d and not 2d, there's no way to know if they're row or column vectors.
				// for now, we just take the column*row interpretation which gives us a matrix.

				// of course assume a is a column vector and b is a row vector.
				// we need to replicate a horizontally, n times

				auto&& u = xt::eval(xt::repeat(xt::atleast_2d(a), bn, 0));
				auto&& v = xt::eval(xt::repeat(xt::atleast_2d(b), an, 0));

				// std::cout << "a: " << xt::adapt(a.shape()) << ",  b: " << xt::adapt(b.shape()) << "\n";
				// std::cout << "u: " << xt::adapt(u.shape()) << ",  v: " << xt::adapt(v.shape()) << "\n";

				auto&& ut = xt::transpose(u);
				auto&& ret = ut * v;

				// std::cout << "a: " << xt::adapt(a.shape()) << ",  b: " << xt::adapt(b.shape()) << ",  ret = " << ret << "\n";
				return xt::eval(ret);
			}
		}
	}


	template <size_t... Ns>
	struct shape
	{
		static constexpr size_t dims = sizeof...(Ns);
		static constexpr std::array<size_t, dims> sizes = { Ns... };

		template <size_t... Is>
		static shape<sizes[Is]...> f1(std::index_sequence<Is...>) { return { }; }

		template <size_t K, size_t... Is>
		static shape<sizes[Is]..., K> f2(std::index_sequence<Is...>) { return { }; }

		template <size_t N, size_t D = dims, typename E = std::enable_if_t<(D >= N)>>
		using drop = decltype(f1(std::make_index_sequence<D - N>()));

		template <size_t N>
		using add = decltype(f2<N>(std::make_index_sequence<dims>()));

		template <size_t D = dims, typename E = std::enable_if_t<(D > 0)>>
		static constexpr size_t last = sizes[D - 1];
	};

	template <typename T1, typename T2, typename NT1, typename NT2, NT1 N1, NT2 N2>
	bool operator== (const xt::const_array<T1, N1>& a, const xt::svector<T2, N2>& b)
	{
		return std::equal(a.begin(), a.end(), b.begin(), b.end());
	}

	template <typename T1, typename T2, typename NT1, typename NT2, NT1 N1, NT2... N2>
	bool operator== (const xt::sequence_view<T2, N2...>& a, const xt::const_array<T1, N1>& b)
	{
		return std::equal(a.begin(), a.end(), b.begin(), b.end());
	}

	template <typename T1, typename T2, typename NT1, typename NT2, NT1 N1, NT2... N2>
	bool operator== (const xt::sequence_view<T2, N2...>& a, const xt::svector<T1, N1>& b)
	{
		return std::equal(a.begin(), a.end(), b.begin(), b.end());
	}




	template <typename T1, size_t N1, typename T2, size_t N2>
	bool operator== (const xt::svector<T2, N2>& a, const xt::const_array<T1, N1>& b)
	{
		return (b == a);
	}

	template <typename T1, typename T2, size_t N1, size_t... N2>
	bool operator== (const xt::svector<T1, N1>& a, const xt::sequence_view<T2, N2...>& b)
	{
		return (b == a);
	}

	template <typename T1, typename T2, size_t N1, size_t... N2>
	bool operator== (const xt::const_array<T1, N1>& a, const xt::sequence_view<T2, N2...>& b)
	{
		return (b == a);
	}



	template <typename T1, size_t N1, typename T2, size_t N2>
	bool operator!= (const xt::const_array<T1, N1>& a, const xt::svector<T2, N2>& b) { return !(a == b); }

	template <typename T1, size_t N1, typename T2, size_t N2>
	bool operator!= (const xt::svector<T2, N2>& a, const xt::const_array<T1, N1>& b) { return !(b == a); }
}
