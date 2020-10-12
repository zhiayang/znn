// util.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "precompile.h"

namespace znn
{
	using xarr = xt::xarray<double>;

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

		template <typename At, typename Bt, typename R = std::common_type_t<typename At::value_type, typename Bt::value_type>>
		std::vector<size_t> result_dims(const xt::xexpression<At>& aexp, const xt::xexpression<Bt>& bexp)
		{
			auto&& a = xt::view_eval<At::static_layout>(aexp.derived_cast());
			auto&& b = xt::view_eval<Bt::static_layout>(bexp.derived_cast());

			std::vector<size_t> out_shape;
			for(size_t i = 0; i < a.dimension() - 1; i++)
				out_shape.push_back(a.shape()[i]);

			for(size_t i = 1; i < b.dimension(); i++)
				out_shape.push_back(b.shape()[i]);

			return out_shape;
		}


		// similar to python matmul -- treats dimension > 2 as a stack of matrices.
		// for two vectors, takes the outer product and returns a matrix.
		// template <class _Tp = double, typename XC1, typename XC2>
		template <typename At, typename Bt, typename R = std::common_type_t<typename At::value_type, typename Bt::value_type>>
		xt::xarray<R> matrix_mul(const xt::xexpression<At>& aexp, const xt::xexpression<Bt>& bexp)
		{
			auto&& a = xt::view_eval<At::static_layout>(aexp.derived_cast());
			auto&& b = xt::view_eval<Bt::static_layout>(bexp.derived_cast());

			using Arr = xt::xarray<double>;

			if(a.dimension() == 1 && b.dimension() == 1)
			{
				return xt::linalg::outer(a, b);
			}
			else if(a.dimension() <= 2 && b.dimension() <= 2)
			{
				return xt::linalg::dot(a, b);
			}
			else
			{
				if(a.dimension() == b.dimension())
				{
					assert(a.shape()[0] == b.shape()[0]);
					size_t layers = a.shape()[0];

					Arr tmp;
					{
						Arr a0 = xt::view(a, 0);
						Arr b0 = xt::view(b, 0);
						tmp = matrix_mul(std::move(a0), std::move(b0));
					}

					auto out_shape = tmp.shape();
					out_shape.insert(out_shape.begin(), layers);

					auto result = Arr::from_shape(out_shape);
					xt::view(result, 0) = tmp;

					for(size_t i = 1; i < layers; i++)
					{
						Arr ai = xt::view(a, i);
						Arr bi = xt::view(b, i);
						xt::view(result, i) = matrix_mul(std::move(ai), std::move(bi));
					}

					return result;
				}
				else if(a.dimension() > b.dimension())
				{
					// for now, let's not support broadcasting more than one dimension at once.
					assert(a.dimension() - 1 == b.dimension());
					size_t layers = a.shape()[0];

					Arr tmp;
					{
						Arr a0 = xt::view(a, 0);
						tmp = matrix_mul(std::move(a0), b);
					}

					auto out_shape = tmp.shape();
					out_shape.insert(out_shape.begin(), layers);

					auto result = Arr::from_shape(out_shape);
					xt::view(result, 0) = std::move(tmp);

					for(size_t i = 1; i < layers; i++)
					{
						Arr ai = xt::view(a, i);
						xt::view(result, i) = matrix_mul(std::move(ai), b);
					}

					return result;
				}
				else
				{
					assert(a.dimension() < b.dimension());

					// for now, let's not support broadcasting more than one dimension at once.
					assert(a.dimension() + 1 == b.dimension());

					size_t layers = b.shape()[0];

					Arr tmp;
					{
						Arr b0 = xt::strided_view(b, { xt::ellipsis(), 0, xt::all(), xt::all() });
						tmp = matrix_mul(a, std::move(b0));
					}

					auto out_shape = tmp.shape();
					out_shape.insert(out_shape.begin(), layers);

					auto result = Arr::from_shape(out_shape);
					xt::strided_view(result, { xt::ellipsis(), 0, xt::all(), xt::all() }) = std::move(tmp);

					for(size_t i = 1; i < layers; i++)
					{
						Arr bi = xt::strided_view(b, { xt::ellipsis(), i, xt::all(), xt::all() });
						xt::strided_view(result, { xt::ellipsis(), i, xt::all(), xt::all() })
							= matrix_mul(a, std::move(bi));
					}

					return result;
				}
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

		static constexpr size_t flatten()
		{
			size_t i = 1;
			for(size_t k : sizes)
				i *= k;

			return i;
		}
	};
}
