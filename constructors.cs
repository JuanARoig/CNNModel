using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet{

	class ConvolutionalNN
	{
		public ConvolutionalNN ConvolutionalNN() { }
		public ConvolutionalNN ConvolutionalNN(int[,] kernelSizes)//kernelSizes j-hat sizes must be of size 3.
		{
			this.kernelSizes = kernelSizes;
		}	
	}
}