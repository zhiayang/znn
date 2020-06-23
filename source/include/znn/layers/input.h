// input.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../misc.h"

namespace znn::layers
{
	namespace impl
	{
		struct InputLayer
		{
			virtual void feed(const xarr& input) = 0;
		};

		template <typename InputShape>
		struct Input : Layer, InputLayer
		{
			Input() : Layer(nullptr) { }

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");
			using OutputShape = InputShape;


			virtual std::pair<xarr, xarr> backward(const xarr&) override
			{
				// do nothing, we have no weights.
				return { };
			}

			virtual void feed(const xarr& input) override
			{
				assert(zfu::equal(input.shape(), InputShape::sizes));
				this->last_output = input;
			}

			virtual xarr compute(bool) override
			{
				return this->last_output;
			}

			virtual void updateWeights(const xarr&, const xarr&, double) override
			{
			}

		private:
			xarr input;
		};
	}

	template <typename InputShape>
	impl::Input<InputShape> Input()
	{
		return impl::Input<InputShape>();
	}
}








