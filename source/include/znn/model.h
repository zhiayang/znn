// model.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "util.h"

namespace znn
{
	struct Model
	{
		Model(layers::impl::InputLayer& input_layer, Layer& output_layer)
			: input_layer(input_layer), output_layer(output_layer) { }

		xarr predict(const xarr& in)
		{
			this->input_layer.feed(in);
			return xt::eval(this->output_layer.compute(/* training: */ false));
		}

		xarr train_forward(const xarr& in)
		{
			this->input_layer.feed(in);
			return xt::eval(this->output_layer.compute(/* training: */ true));
		}

		Layer* outputLayer()  { return &output_layer; }

	private:
		layers::impl::InputLayer& input_layer;
		Layer& output_layer;
	};
}
