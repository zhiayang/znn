// znn.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "precompile.h"

#include "misc.h"
#include "layers.h"
#include "optimisers.h"
#include "activations.h"
#include "regularisers.h"

namespace znn
{
	template <typename Opt>
	void train(Model& model, const std::vector<xarr>& x, const std::vector<xarr>& y, Opt& optimiser)
	{
		optimiser.run(model, x, y);
	}
}
