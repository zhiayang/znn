// activation.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "misc.h"

namespace znn::activations
{
	struct Activation
	{
	};

	struct None : Activation
	{
		static xarr forward(const xarr& input)
		{
			return input;
		}

		static xarr derivative(const xarr& input)
		{
			return xt::ones<double>(input.shape());
		}
	};

	struct ReLU : Activation
	{
		static xarr forward(const xarr& input)
		{
			return xt::where(input <= 0, xt::zeros<double>(input.shape()), input);
		}

		static xarr derivative(const xarr& input)
		{
			return xt::where(input < 0, xt::zeros<double>(input.shape()), xt::ones<double>(input.shape()));
		}
	};

	struct Sigmoid : Activation
	{
		static xarr forward(const xarr& input)
		{
			return 1.0 / (1.0 + xt::exp(-input));
		}

		static xarr derivative(const xarr& input)
		{
			return input * (1 - input);
		}
	};

	struct TanH : Activation
	{
		static xarr forward(const xarr& input)
		{
			return xt::tanh(input);
		}

		static xarr derivative(const xarr& input)
		{
			return 1.0 - xt::pow(input, 2);
		}
	};
}
