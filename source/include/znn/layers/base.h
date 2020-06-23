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

		virtual xarr compute(bool training) = 0;
		virtual void backward(const xarr& err) = 0;
		virtual void updateWeights(optimisers::Optimiser* opt, double scale) = 0;

		const xarr& getLastOutput() { return this->last_output; }

		Layer* prev() { return input_layer; }

		void resetDeltas()
		{
			this->d_weight = xt::zeros<double>(this->d_weight.shape());
			this->d_bias = xt::zeros<double>(this->d_bias.shape());

			if(this->prev())
				this->prev()->resetDeltas();
		}

	protected:
		Layer(Layer* in) : input_layer(in) { }
		xarr last_output = { };
		xarr d_weight = { };
		xarr d_bias = { };

		void update_dw_db(const xarr& gradient)
		{
			this->d_weight += util::matrix_mul(xt::transpose(gradient), this->prev()->getLastOutput());
			this->d_bias += gradient;
		}

	private:
		Layer* input_layer = 0;
	};
}
