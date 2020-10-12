// base.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../util.h"

namespace znn
{
	struct Layer;
	namespace optimisers
	{
		struct Optimiser
		{
			virtual ~Optimiser() { }
			virtual void computeDeltas(Layer* layer, xarr& dw, xarr& db) = 0;
		};
	}

	struct Layer
	{
		virtual ~Layer() { }

		virtual xarr compute(bool training, bool batched) = 0;
		virtual void backward(const xarr& err, bool batched) = 0;
		virtual void updateWeights(optimisers::Optimiser* opt, double scale) = 0;

		const xarr& getLastOutput() { return this->last_output; }

		Layer* prev() { assert(input_layer); return input_layer; }

		void resetDeltas()
		{
			this->d_weight = xt::zeros<double>(this->d_weight.shape());
			this->d_bias = xt::zeros<double>(this->d_bias.shape());

			if(this->input_layer != nullptr)
				this->input_layer->resetDeltas();
		}

	protected:
		Layer(Layer* in) : input_layer(in) { }
		xarr last_output = { };
		xarr d_weight = { };
		xarr d_bias = { };

		template <typename InputShape>
		bool ensure_correct_dimensions(const xarr& input, bool batched)
		{
			if(batched)
			{
				return (input.dimension() == InputShape::dims + 1)
					&& (std::equal(input.shape().begin() + 1, input.shape().end(),
									InputShape::sizes.begin(), InputShape::sizes.end()));
			}
			else
			{
				return zfu::equal(input.shape(), InputShape::sizes);
			}
		}

		auto unbatched_input_shape(const xarr& input, bool batched)
		{
			auto shape = input.shape();
			using shape_t = decltype(shape);

			return shape_t(shape.begin() + (batched ? 1 : 0), shape.end());
		}

	private:
		Layer* input_layer = 0;
	};
}
