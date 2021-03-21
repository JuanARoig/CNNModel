using System;
using System.Collections.Generic;

namespace neuralnet{

	public partial class ConvolutionalNN
	{
		public ConvolutionalNN() { }
		
		public ConvolutionalNN(double[,,] inputData, int[,] kernelSizes, int transitionElementAmount, int finalOutputSize, int seed)
		{
			if (kernelSizes.GetLength(1) > 3)
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
			this.convLayerNetwork = new List<double[,,]>();
			for (int i = 0; i < kernelSizes.Length; i++)
			{
				this.convLayerKernels.Add(new double[kernelSizes[i, 0], kernelSizes[i, 1], kernelSizes[i, 2]]);
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
				double[,,] kernel = this.convLayerKernels[kernelIndex];
				if (sizeI - kernel.GetLength(0) == 1) 
	    		{
	    			sizeFinalI += 2;
	    		} 
	    		else 
	    		{
	    			sizeFinalI += (-kernel.GetLength(0)) + 1;
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
			//popping unnecessary kernels
			if(kernelIndex < convLayerKernels.Capacity)
			{
				for (int i = kernelIndex; i < convLayerKernels.Capacity; i++)
				{
					convLayerKernels.RemoveAt(i);
				}
			}
			
			this.transitionElementAmount = sizeFinalI*sizeFinalJ*sizeFinalK;
			int elementCountChange = (this.transitionElementAmount - this.finalOutputSize)/this.FCLayerAmount;
			this.FCLayerSizes = new int[this.FCLayerAmount];
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
			for (int i = 1; i < this.FCLayerSizes.Length; i++)
			{
				this.weights.Add(new double[this.FCLayerSizes[i],this.FCLayerSizes[i - 1]]);
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
			this.convLayerKernelUpdates = new List<double[,,]>();
			this.weightsupdates = new List<double[,]>();
			this.biasupdates = new List<double[]>();
			for(int x = 0; x < this.convLayerKernels.Capacity; x++)
			{
				double[,,] currentFormat = this.convLayerKernels[x];
				this.convLayerKernelUpdates.Add(new double[currentFormat.GetLength(0),
													  currentFormat.GetLength(1),
													  currentFormat.GetLength(2)]);	
			}
			for (int x = 0; x < this.weights.Capacity; x++)
			{
				double[,] currentFormat = this.weights[x];
				this.weightsupdates.Add(new double[currentFormat.GetLength(0),
										   currentFormat.GetLength(1)]);
			}
			for (int x = 0; x < this.biases.Capacity; x++)
			{
				double[] currentFormat = this.biases[x];
				this.biasupdates.Add(new double[currentFormat.Length]);
			}	
		}	
	}
}