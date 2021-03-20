using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Xunit;
using neuralnet;

namespace CNNModel.Tests
{
    public class CNNTests : ConvolutionalNN
    {

    	[Theory]
    	public void ConvolutionTest(double[,,] testInputTensor, double[,,] testKernel)
    	{
		
    	}
		[Theory]
		public void AvgPoolingTest(double[,,] testInputTensor, double[] testPrevKernelDimensions)
		{

		}
		[Theory]
		public void MaxPoolingTest(double[,,] testInputTensor, double[] testPrevKernelDimensions)
		{

		}
		public static IEnumerable<double[,,]> GetTestInput()
		{
			double[,,] test = {{{2, 2}}};
			double[,,] test2 = {{{2},{6},{6}},{{6}, {6}, {6}}};
			yield return new double[,,] { test };
			yield return new double[,,] { test2 };
		}
		[Theory]
		[InlineData()]
		public void RElu_Evaluates(double[,,] testInput)
		{
			Assert.True(REluOperation(testInput) != null);
		}
		[Fact]
		public void RElu_Throws_Exception()
		{
			double[,,] test = {{{}}};
			Assert.Throws<Exception>(() => REluOperation(test));
		}
		[Theory]
		public void SoftmaxTest(double[] testInput)
		{
			
		}
		[Theory]
		public void MatrixVectorProductTest(double[] testInput)
		{
			
		}
		[Theory]
		public void AddTest(double[] testInput)
		{
			
		}
		[Theory]
		public void GeneralCNN(double[,,] testInput)
		{

		}
    }
}
