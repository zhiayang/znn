// cost.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "util.h"

namespace znn::cost
{
	struct MeanSquare
	{
		double calculate(const xarr& target, const xarr& prediction)
		{
			assert(target.shape() == prediction.shape());
			return xt::sum(0.5 * xt::square(prediction - target))() / target.size();
		}

		xarr derivative(const xarr& target, const xarr& prediction)
		{
			assert(target.shape() == prediction.shape());
			return (prediction - target);
		}
	};
}
