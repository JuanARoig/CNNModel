using System;

namespace neuralnet
{
    public partial class ConvolutionalNN
    {
        protected double[,,] AvgPoolingOperation(double[,,] inputTensor, int[] prevKernelDimensions) 
        { 
			if (inputTensor.GetLength(0) <= prevKernelDimensions[0] || inputTensor.GetLength(1) <= prevKernelDimensions[1] || inputTensor.GetLength(2) <= prevKernelDimensions[2])
			{
				throw new Exception("Tensor dims collapsed too quickly.");
			}
            int sizeI = 0;
	    	int sizeJ = 0;
	    	int sizeK = 0;

            //the dimensions of the resulting tensor are funky.
	    	if (inputTensor.GetLength(0) - prevKernelDimensions[0] == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (inputTensor.GetLength(0)-prevKernelDimensions[0]) + 1;
	    	}
	    	if (inputTensor.GetLength(1) - prevKernelDimensions[1] == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (inputTensor.GetLength(1) - prevKernelDimensions[1]) + 1;
	    	}
	    	if (inputTensor.GetLength(2) - prevKernelDimensions[2] == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (inputTensor.GetLength(2) - prevKernelDimensions[2]) + 1;
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

            for(int i = 0; i < inputTensor.GetLength(0) - prevKernelDimensions[0] - 1; i++)
	    	{
			    for(int j = 0; j < inputTensor.GetLength(1) - prevKernelDimensions[1] - 1; j++)
		    	{
				    for(int k = 0; k < inputTensor.GetLength(2) - prevKernelDimensions[2] - 1; k++)
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
	    protected double[,,] MaxPoolingOperation(double[,,] inputTensor, int[] prevKernelDimensions) 
        { 
			if (inputTensor.GetLength(0) <= prevKernelDimensions[0] || inputTensor.GetLength(1) <= prevKernelDimensions[1] || inputTensor.GetLength(2) <= prevKernelDimensions[2])
			{
				throw new Exception("Tensor dims collapsed too quickly.");
			}
            int sizeI = 0;
	    	int sizeJ = 0;
	    	int sizeK = 0;

            //the dimensions of the resulting tensor are funky.
	    	if (inputTensor.GetLength(0) - prevKernelDimensions[0] == 1) 
	    	{
	    		sizeI += 2;
	    	} 
	    	else 
	    	{
	    		sizeI += (inputTensor.GetLength(0)-prevKernelDimensions[0]) + 1;
	    	}
	    	if (inputTensor.GetLength(1) - prevKernelDimensions[1] == 1)
	    	{
	    		sizeJ += 2;
	    	}
	    	else
	    	{
	    		sizeJ += (inputTensor.GetLength(1) - prevKernelDimensions[1]) + 1;
	    	}
	    	if (inputTensor.GetLength(2) - prevKernelDimensions[2] == 1)
	    	{
	    		sizeK += 2;
	    	}
	    	else
	    	{
	    		sizeK += (inputTensor.GetLength(2) - prevKernelDimensions[2]) + 1;
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

            for(int i = 0; i < inputTensor.GetLength(0) - prevKernelDimensions[0] - 1; i++)
	    	{
			    for(int j = 0; j < inputTensor.GetLength(1) - prevKernelDimensions[1] - 1; j++)
		    	{
				    for(int k = 0; k < inputTensor.GetLength(2) - prevKernelDimensions[2] - 1; k++)
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