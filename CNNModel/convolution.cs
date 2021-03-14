using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        private double[,,] ConvolutionOperation(double[,,] inputTensor, double[,,] kernel) 
	    {	
			if (inputTensor.GetLength(0) <= kernel.GetLength(0) || inputTensor.GetLength(1) <= kernel.GetLength(1) || inputTensor.GetLength(2) <= kernel.GetLength(2))
			{
				throw new Exception("Tensor dims collapsed too quickly.");
			}
	    	int sizeI = 0;
	    	int sizeJ = 0;
	    	int sizeK = 0;

            //the dimensions of the resulting tensor are funky.
	    	if (inputTensor.GetLength(0) - kernel.GetLength(0) == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (inputTensor.GetLength(0)-kernel.GetLength(0)) + 1;
	    	}
	    	if (inputTensor.GetLength(1) - kernel.GetLength(1) == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (inputTensor.GetLength(1) - kernel.GetLength(1)) + 1;
	    	}
	    	if (inputTensor.GetLength(2) - kernel.GetLength(2) == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (inputTensor.GetLength(2) - kernel.GetLength(2)) + 1;
	    	}

	    	double[,,] tensorFinal = new double[sizeI,sizeJ,sizeK];
		    for (int i = 0; i < tensorFinal.GetLength(0); i++)
	    	{
		    	for(int j = 0; j < tensorFinal.GetLength(1); j++)	
		    	{	
		    		for(int k = 0; k < tensorFinal.GetLength(2); k++)
		    		{
		    			tensorFinal[i, j, k] = 0;
		    		}
		    	}
	    	}
		
	    	for(int i = 0; i < inputTensor.GetLength(0) - kernel.GetLength(0) - 1; i++)
	    	{
			    for(int j = 0; j < inputTensor.GetLength(1) - kernel.GetLength(1) - 1; j++)
		    	{
				    for(int k = 0; k < inputTensor.GetLength(2) - kernel.GetLength(2) - 1; k++)
				    {
					    int startA = i;
					    int startB = j;
					    int startC = k;
					    int r = kernel.GetLength(0);
					    int s = kernel.GetLength(1);
					    int t = kernel.GetLength(2);
					    for(int a = i; a < startA + r; a++)
					    {
					    	for (int b = j; b < startB + s; b++)
						    {
							    for (int c = k; c < startC + t; c++)
						    	{
							    	tensorFinal[i, j, k] += inputTensor[a, b, c]*kernel[a-startA, b-startB, c-startC];
							    }
						    }
					    }				
				    }
			    }
		    }
		    return tensorFinal;	
	    }	
    }
}