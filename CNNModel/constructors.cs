using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes, int transitionElementAmount, int finalOutputSize, int seed)
		{
			if (kernel1Size[0] > 3)
			{
				throw new Exception("this is not a 4d convnet.");
			}
			this.seed = seed;
			Random random = new Random(this.seed);
			this.kernelSizes = kernelSizes;
			this.transitionElementAmount = transitionElementAmount;
			//when element amount falls below the transitionElementAmount, network will transition from
			//convolutional to fully connected.


			this.finalOutputSize = finalOutputSize;
			this.biases = new List<double[]>();
			this.weights = new List<double[,]>();
			this.convLayerKernels = new List<double[,,]>();
			for (int i = 0; i < kernelSizes.Length; i++)
			{
				this.convLayerKernels.Add(new double[kernelSizes[i, 0],kernelSizes[i, 1], kernelSizes[i, 2]);
			}
			foreach(double[,,] kernel in convLayerKernels)
			{
				for (int i = 0; i < kernel.GetLength(0); i++)
				{
					for (int j = 0; j < kernel.GetLength(1); j++)
					{
						for (int k = 0; k < kernel.GetLength(2); k++)
						{
							//make sure this is referencing the same object.
							kernel[i, j, k] = random.NextDouble(this.seed);
						}
					}
				}
			}
		}	
	}
}