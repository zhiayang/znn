// activations.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "misc.h"

namespace znn::activations
{
	struct Activation
	{
	};

	// in all of these cases, the derivative takes in the input *AFTER* it has already gone
	// through the forward pass. so for example, d/dx (sigmoid) is sigmoid(x) * (1-sigmoid(x))
	// but we compute [x * (1-x)], since x is already sigmoided.

	struct Linear : Activation
	{
		xarr forward(const xarr& input)
		{
			return input;
		}

		xarr derivative(const xarr& input)
		{
			return xt::ones<double>(input.shape());
		}
	};

	struct ReLU : Activation
	{
		xarr forward(const xarr& input)
		{
			return xt::where(input <= 0, xt::zeros<double>(input.shape()), input);
		}

		xarr derivative(const xarr& input)
		{
			return xt::where(input <= 0, xt::zeros<double>(input.shape()), xt::ones<double>(input.shape()));
		}
	};

	struct Sigmoid : Activation
	{
		xarr forward(const xarr& input)
		{
			return 1.0 / (1.0 + xt::exp(-input));
		}

		xarr derivative(const xarr& input)
		{
			return input * (1 - input);
		}
	};

	struct TanH : Activation
	{
		xarr forward(const xarr& input)
		{
			return xt::tanh(input);
		}

		xarr derivative(const xarr& input)
		{
			return 1.0 - xt::square(input);
		}
	};
}
