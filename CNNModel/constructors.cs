using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	partial class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }

		public ConvolutionalNN ConvolutionalNN(double[,,] inputData, int[,] kernelSizes, int transitionElementAmount, int finalOutputSize, int seed)
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
			this.convLayerNetwork = new List<double,,]>();
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
							//by reference, should work.
							kernel[i, j, k] = random.NextDouble();
						}
					}
				}
			}
			int kernelIndex = 0;
			int sizeI = inputData.GetLength(0);
			int sizeJ = inputData.GetLength(1);
			int sizeK = inputData.GetLength(2);
			int elems = inputData.GetLength(0)*inputData.GetLength(1)*inputData.GetLength(2);
			int sizeFinalI = 0;
			int sizeFinalJ = 0;
			int sizeFinalK = 0;
			while(elems < this.transitionElementAmount && kernelIndex < convLayerKernels.Capacity)
			{
				kernel = this.convLayerKernels[kernelIndex];
				if (sizeI - kernel.GetLength(0) == 1) 
	    		{
	    			sizeFinalI += 2;
	    		} 
	    		else 
	    		{
	    			sizeFinalI += (-kernel.GetLength(0)) + 1;sizeI
	    		}
	    		if (sizeJ - kernel.GetLength(1) == 1)
	    		{
	    			sizeFinalJ += 2;
	    		}
	    		else
	    		{
	    			sizeJ += (sizeJ - kernel.GetLength(1)) + 1;
	    		}
	    		if (sizeK - kernel.GetLength(2) == 1)
	    		{
	    			sizeFinalK += 2;
	    		}
	    		else
	    		{
	    			sizeFinalK += (sizeK - kernel.GetLength(2)) + 1;
	    		}
				elems = sizeFinalI*sizeFinalJ*sizeFinalK;
				kernelIndex++;
			}
			this.transitionElementAmount = sizeFinalI*sizeFinalJ*sizeFinalK;
			//a clean relationship between transitionElementAmount and finalOutputSize
			//may not be possible in some cases. Draw it out on paper before continuing.
			if ((this.transitionElementAmount - this.finalOutputSize - 1) % this.FCLayerAmount == 0)
			{
				int elementCountChange = (this.transitionElementAmount - this.finalOutputSize)/this.FCLayerAmount;
				this.FCLayerSizes = new double[this.FCLayerAmount];
				this.FCLayerSizes[0] = this.transitionElementAmount;
				this.FCLayerSizes[this.FCLayerSizes.Length - 1] = this.finalOutputSize;
				int startSize = this.transitionElementAmount;
				for (int i = 1; i < this.FCLayerSizes.Length - 1; i++)
				{
					startSize -= elementCountChange;
					this.FCLayerSizes[i] = startSize;
				}
				for (int i = 1; i < this.FCLayerSizes.Length; i++)
				{
					this.biases.Add(new double[i]);
				}
				for (int i = 1, j = 0; i++, j++; i < this.FCLayerSizes.Length, j < this.FCLayerSizes.Length-1)
				{
					this.weights.Add(new double[this.FCLayerSizes[i],this.FCLayerSizes[j]]);
				}
				foreach (double[] biasVector in this.biases)
				{
					for (int i = 0; i < biasVector.Length; i++)
					{
						biasVector[i] = random.NextDouble();
					}
				}
				foreach (double[,] weightsMatrix in this.weights)
				{
					for (int i = 0; i < weightsMatrix.GetLength(0); i++)
					{
						for (int j = 0; j < weightsMatrix.GetLength(1); j++)
						{
							weightsMatrix[i, j] = random.NextDouble();
						}
					}
				}
			}
			else
			{
				
			}
		}	
	}
}