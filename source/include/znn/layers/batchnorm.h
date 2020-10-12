// batchnorm.h
// Copyright (c) 2020, zhiayang
// Licensed under the Apache License Version 2.0.

#pragma once

#include "../util.h"
#include "../activations.h"
#include "../regularisers.h"

namespace znn::layers
{
	namespace impl
	{
		/*
			since batchnorm is usually used before the activation, the layers are usually structured
			like Dense<Linear> -> BatchNorm -> Activation<AF>
			so we let people simplify it by letting batchnorm do the activation, so it just becomes
			Dense<Linear> -> BatchNorm<AF>

			about "channelled":
			certain 2d inputs (eg. colour images) are characterised by a dimension of CxWxH, where C
			is the number of channels (eg. 3 for rgb). batchnorm is supposed to normalise each channel
			individually, so the channels don't influence each other.

			we might also want to support arbitrary dimensional batches; for example, a 4-d input
			could be (Batch, Channel, W, H) for a channelled 2d-input, or it could be a mono-channel
			(Batch, D, W, H) 3d-input. so we just specify whether there's a channel. in the case of
			channelled input, the mean and variance are calculated per channel (meaning they are vectors
			of length C). for un-channelled input, they are simply scalars.
		*/
		template <typename InputLayer, typename ActivationFn, bool Channelled>
		struct BatchNorm : Layer
		{
			BatchNorm(InputLayer& input, double momentum, double epsilon, ActivationFn af)
				: Layer(&input), momentum(momentum), epsilon(epsilon), activator(std::move(af))
			{
				assert(epsilon > 0);
				assert(0 < momentum && momentum <= 1);

				this->axes = std::vector<size_t>(InputShape::dims);
				std::iota(this->axes.begin(), this->axes.end(), 0);

				this->batchedAxes = std::vector<size_t>(1 + InputShape::dims);
				std::iota(this->batchedAxes.begin(), this->batchedAxes.end(), 0);

				if constexpr (Channelled)
				{
					this->axes.erase(this->axes.begin());
					this->batchedAxes.erase(this->batchedAxes.begin() + 1);
				}
			}

			using InputShape = typename InputLayer::OutputShape;
			using OutputShape = InputShape;

			static_assert(InputShape::dims > 0, "input shape cannot be 0-dimensional");


			virtual xarr compute(bool training, bool batched) override
			{
				auto input = this->prev()->compute(training, batched);
				assert(ensure_correct_dimensions<InputShape>(input, batched));

				xarr* mu = nullptr;
				xarr* sigma = nullptr;

				if(training)
				{
					if(batched)
					{
						// std::cout << xt::adapt(xt::mean(input, axes).shape()) << "\n";

						this->mean   = xt::mean(input, this->batchedAxes);
						this->variance = xt::variance(input, this->batchedAxes);
					}
					else
					{
						if constexpr (Channelled)
						{
							this->mean   = xt::mean(input, this->axes);
							this->variance = xt::variance(input, this->axes);
						}
						else
						{
							this->mean   = { xt::mean(input) };
							this->variance = { xt::variance(input) };
						}
					}

					this->movingMean = (this->momentum * this->mean) + (1 - this->momentum) * this->movingMean;
					this->movingVariance = (this->momentum * this->variance) + (1 - this->momentum) * this->movingVariance;

					mu = &this->mean;
					sigma = &this->variance;
				}
				else
				{
					mu = &this->movingMean;
					sigma = &this->movingVariance;
				}

				auto&& output = /*this->beta + this->gamma * */((input - *mu) / xt::sqrt(*sigma + this->epsilon));

				this->last_output = std::move(output);
				return this->last_output;
			}

			virtual void backward(const xarr& error, bool batched) override
			{
				assert(ensure_correct_dimensions<OutputShape>(error, batched));
				auto&& input = this->prev()->getLastOutput();

				auto&& stddev_inv = (1.0 / xt::sqrt(this->variance + this->epsilon));

				/*
					∂L/∂x̂  = ∂L/∂y * γ
					∂L/∂σ² = Σ[∂L/∂x̂ * (x - μ) * -0.5(σ² + ε)^(-3/2)]
					∂L/∂μ  = Σ[∂L/∂x̂ * -(1/√[σ² + ε])] + (∂L/∂σ² * 1/m (-2Σ[x - μ]))

					∂L/∂x  = ∂L/∂x̂ * 1/√[σ² + ε] + ∂L/∂σ² * 2(x - μ)/m + ∂L/∂μ * 1/m
					∂L/∂γ  = Σ[∂L/∂y * x̂]
					∂L/∂β  = Σ[∂L/∂y]
				*/

				double batch_size = (batched ? error.shape()[0] : 1);

				auto& axis = batched ? this->batchedAxes : this->axes;
				auto&& diff = input - this->mean;

				auto&& d_norm = error * this->gamma;

				// auto&& d_var  = xt::sum(d_norm * diff, axis) * -0.5 * xt::pow(stddev_inv, 3);
				// auto&& d_x    = (d_norm * stddev_inv) + (diff * (d_var * 2 / batch_size))
				// 				+ xt::sum(d_norm * -stddev_inv, axis) / batch_size;



				auto&& d_var  = xt::sum(d_norm * diff * -0.5 * xt::pow(stddev_inv, 3), axis);
				auto&& d_mean = xt::sum(d_norm * -stddev_inv, axis) + (d_var * xt::mean(-2 * diff, axis));
				auto&& d_x    = (d_norm * stddev_inv) + (d_var * 2 * diff / batch_size) + d_mean / batch_size;


			/*
				auto&& d_x    = stddev_inv * (d_norm - xt::mean(d_norm, axis))
								- this->last_output * xt::mean(d_norm * this->last_output, axis);
			*/
				auto&& d_gamm = xt::sum(error * this->last_output, axis);
				auto&& d_beta = xt::sum(error, axis);

				this->d_weight += d_gamm;
				this->d_bias   += d_beta;

				this->prev()->backward(d_x, batched);
			}

			virtual void updateWeights(optimisers::Optimiser* opt, double scale) override
			{
				this->gamma -= scale * this->d_weight;
				this->beta  -= scale * this->d_bias;

				this->prev()->updateWeights(opt, scale);
			}

		private:
			const double momentum = 0;
			const double epsilon = 0;
			ActivationFn activator;

			xarr gamma = { 1 };
			xarr beta = { 0 };

			// this is the per-batch stuff
			xarr mean;
			xarr variance;

			// this is the moving mean/stddev to represent the entire dataset.
			// we use this when predicting (not training)
			xarr movingMean;
			xarr movingVariance;

			std::vector<size_t> axes;
			std::vector<size_t> batchedAxes;
		};
	}

	template <typename InputLayer, typename AF = activations::Linear>
	impl::BatchNorm<InputLayer, AF, false> BatchNorm(InputLayer& il,
		double momentum, double epsilon = 1e-8)
	{
		return impl::BatchNorm<InputLayer, AF, false>(il, momentum, epsilon, AF());
	}

	template <typename InputLayer, typename AF = activations::Linear>
	impl::BatchNorm<InputLayer, AF, false> BatchNorm(InputLayer& il, const AF& af = AF(),
		double momentum = 0.999, double epsilon = 1e-8)
	{
		return impl::BatchNorm<InputLayer, AF, false>(il, momentum, epsilon, af);
	}

	template <typename InputLayer, typename AF = activations::Linear>
	impl::BatchNorm<InputLayer, AF, true> BatchChannelNorm(InputLayer& il,
		double momentum, double epsilon = 1e-8)
	{
		return impl::BatchNorm<InputLayer, AF, true>(il, momentum, epsilon, AF());
	}

	template <typename InputLayer, typename AF = activations::Linear>
	impl::BatchNorm<InputLayer, AF, true> BatchChannelNorm(InputLayer& il, const AF& af = AF(),
		double momentum = 0.999, double epsilon = 1e-8)
	{
		return impl::BatchNorm<InputLayer, AF, true>(il, momentum, epsilon, af);
	}
}
