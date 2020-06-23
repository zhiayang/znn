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

		virtual xarr compute(bool training) = 0;
		virtual std::pair<xarr, xarr> backward(const xarr& del) = 0;
		virtual void updateWeights(const xarr& dw, const xarr& db, double scale) = 0;

		const xarr& getLastOutput() { return this->last_output; }

		Layer* prev() { return input_layer; }

	protected:
		Layer(Layer* in) : input_layer(in) { }
		xarr last_output = { };

	private:
		Layer* input_layer = 0;
	};

	namespace util
	{
		struct __random_state_t
		{
			bool is_fixed = false;
			uint32_t fixed_seed = 0;
			std::random_device rd;
		};

		// note: static-duration variables in external-linkage functions will be collapsed
		// to refer to the same storage.
		inline __random_state_t& __get_global_random_state()
		{
			static __random_state_t state;
			return state;
		}

		static inline uint32_t getSeed()
		{
			auto& st = __get_global_random_state();
			if(st.is_fixed) return st.fixed_seed;
			else            return st.rd();
		}

		static inline void setSeed(uint32_t value)
		{
			auto& st = __get_global_random_state();
			st.fixed_seed = value;
			st.is_fixed = true;

			// also set the seed for xt
			xt::random::seed(value);
		}

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

				// std::cout << "a: " << xt::adapt(a.shape()) << ",  b: " << xt::adapt(b.shape()) << "\n";

				auto&& u = xt::eval(xt::repeat(xt::atleast_2d(a), bn, 0));
				auto&& v = xt::eval(xt::repeat(xt::atleast_2d(b), an, 0));

				// std::cout << "u: " << xt::adapt(u.shape()) << ",  v: " << xt::adapt(v.shape()) << "\n";

				auto&& ut = xt::transpose(u);
				auto&& ret = ut * v;

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
}
