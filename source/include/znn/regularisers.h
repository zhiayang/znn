// regularisers.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "util.h"

namespace znn::regularisers
{
	struct None
	{
		xarr forward(const xarr& weights)
		{
			return xt::zeros<double>(weights.shape());
		}

		xarr derivative(const xarr& weights)
		{
			return xt::zeros<double>(weights.shape());
		}
	};

	struct L1
	{
		L1() = delete;
		L1(double lambda) : lambda(lambda) { }

		xarr forward(const xarr& weights)
		{
			return 0.5 * lambda * xt::abs(weights);
		}

		xarr derivative(const xarr& weights)
		{
			return lambda * xt::sign(weights);
		}

	private:
		double lambda;
	};

	struct L2
	{
		L2() = delete;
		L2(double lambda) : lambda(lambda) { }

		xarr forward(const xarr& weights)
		{
			return 0.5 * lambda * xt::square(weights);
		}

		xarr derivative(const xarr& weights)
		{
			return lambda * weights;
		}

	private:
		double lambda;
	};
}
