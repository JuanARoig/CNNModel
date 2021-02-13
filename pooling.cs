using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace neuralnet
{
    partial class ConvolutionalNN
    {
        private double[,,] AvgPoolingOperation(double[,,] inputTensor, double[] prevKernelDimensions) 
        { 
            sizeI = 0;
	    	sizeJ = 0;
	    	sizeK = 0;

            //the dimensions of the resulting tensor are funky.
	    	if (tensor1.GetLength(0) - prevKernelDimensions[0] == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (tensor1.GetLength(0)-prevKernelDimensions[0]) + 1;
	    	}
	    	if (tensor1.GetLength(1) - prevKernelDimensions[1] == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (tensor1.GetLength(1) - prevKernelDimensions[1]) + 1;
	    	}
	    	if (tensor1.GetLength(2) - prevKernelDimensions[2] == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (tensor1.GetLength(2) - prevKernelDimensions[2]) + 1;
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

            for(int i = 0; i < tensor1.GetLength(0) - prevKernelDimensions[0] - 1; i++)
	    	{
			    for(int j = 0; j < tensor1.GetLength(1) - prevKernelDimensions[1] - 1; j++)
		    	{
				    for(int k = 0; k < tensor1.GetLength(2) - prevKernelDimensions[2] - 1; k++)
				    {
					    int startA = i;
					    int startB = j;
					    int startC = k;
					    int r = prevKernelDimensions[0];
					    int s = prevKernelDimensions[1];
					    int t = prevKernelDimensions[2];
                        double divisor = prevKernelDimensions[0]*prevKernelDimensions[1]*prevKernelDimensions[2];
					    for(int a = i; a < startA + r; a++)
					    {
					    	for (int b = j; b < startB + s; b++)
						    {
							    for (int c = k; c < startC + t; c++)
						    	{
							    	tensorFinal[i, j, k] += inputTensor[i, j, k];
							    }
						    }
					    }
                        tensorFinal[i, j, k] /= divisor;				
				    }
			    }
		    }
		    return tensorFinal;
        }
	    private double[,,] MaxPoolingOperation(double[,,] inputTensor, double[] prevKernelDimensions) 
        { 
            sizeI = 0;
	    	sizeJ = 0;
	    	sizeK = 0;

            //the dimensions of the resulting tensor are funky.
	    	if (tensor1.GetLength(0) - prevKernelDimensions[0] == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (tensor1.GetLength(0)-prevKernelDimensions[0]) + 1;
	    	}
	    	if (tensor1.GetLength(1) - prevKernelDimensions[1] == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (tensor1.GetLength(1) - prevKernelDimensions[1]) + 1;
	    	}
	    	if (tensor1.GetLength(2) - prevKernelDimensions[2] == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (tensor1.GetLength(2) - prevKernelDimensions[2]) + 1;
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

            for(int i = 0; i < tensor1.GetLength(0) - prevKernelDimensions[0] - 1; i++)
	    	{
			    for(int j = 0; j < tensor1.GetLength(1) - prevKernelDimensions[1] - 1; j++)
		    	{
				    for(int k = 0; k < tensor1.GetLength(2) - prevKernelDimensions[2] - 1; k++)
				    {
					    int startA = i;
					    int startB = j;
					    int startC = k;
					    int r = prevKernelDimensions[0];
					    int s = prevKernelDimensions[1];
					    int t = prevKernelDimensions[2];
					    for(int a = i; a < startA + r; a++)
					    {
					    	for (int b = j; b < startB + s; b++)
						    {
							    for (int c = k; c < startC + t; c++)
						    	{
							    	if (tensorFinal[i, j, k] < inputTensor[a, b, c])
                                    {
                                        tensorFinal[i, j, k] = inputTensor[a, b, c];
                                    }
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