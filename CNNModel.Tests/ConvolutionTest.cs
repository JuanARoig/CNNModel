using System;
using Xunit;

namespace CNNModel.Tests
{
    public class CNNTests : CNNModel
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
			yield return new object[] { test };
			yield return new object[] { test2 };
		}
		[Theory]
		[MemberData(nameof(GetTestInput()))]
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
		[MemberData(nameof()]
		public void GeneralCNN(double[,,] testInput)
		{

		}
    }
}
