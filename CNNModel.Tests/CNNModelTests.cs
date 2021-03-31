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
    	[Fact]
    	public void Convolution_Evaluates_Input1()
    	{
			//fill this.
			Assert.True(ConvolutionOperation(new double[,,]{{{}}}, new double[,,] {{{}}}) != null);
    	}
		[Fact]
    	public void Convolution_Evaluates_Input2()
    	{
			//fill this.
			Assert.True(ConvolutionOperation(new double[,,]{{{}}}, new double[,,] {{{}}}) != null);
    	}
		[Fact]
		public void Convolution_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => ConvolutionOperation(new double[,,] {{{}}}, new double[,,] {{{}}}));
		}
		[Fact]
		public void AvgPooling_Evaluates_Input1()
		{
			//fill this.
			Assert.True(AvgPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null);
		}
		[Fact]
		public void AvgPooling_Evaluates_Input2()
		{
			//fill this.
			Assert.True(AvgPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null);
		}
		[Fact]
		public void AvgPooling_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => (AvgPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null));
		}
		[Fact]
		public void MaxPooling_Evaluates_Input1()
		{
			//fill this.
			Assert.True(MaxPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null);
		}
		[Fact]
		public void MaxPooling_Evaluates_Input2()
		{
			//fill this.
			Assert.True(MaxPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null);
		}
		[Fact]
		public void MaxPooling_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => (MaxPoolingOperation(new double[,,] {{{}}}, new int[] {}) != null));			
		}
		[Fact]
		public void RElu_Evaluates_Input1()
		{
			Assert.True(REluOperation(new double[,,] {{{2, 2}}}) != null);
		}
		[Fact]
		public void RElu_Evaluates_Input2()
		{
			Assert.True(REluOperation(new double[,,] {{{2},{6},{6}},{{6}, {6}, {6}}}) != null);
		}
		[Fact]
		public void RElu_Throws_Exception()
		{
			Assert.Throws<Exception>(() => REluOperation(new double[,,] {{{}}}));
		}
		[Fact]
		public void SoftmaxEvaluates_Input1()
		{
			//fill this.
			Assert.True(SoftmaxOperation(new double[] {}) != null);			
		}
		[Fact]
		public void SoftmaxEvaluates_Input2()
		{
			//fill this.
			Assert.True(SoftmaxOperation(new double[] {}) != null);			
		}
		[Fact]
		public void Softmax_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => SoftmaxOperation(new double[] {}));
		}
		[Fact]
		public void MatrixVectorProduct_Evaluates_Input1()
		{
			//fill this.
			Assert.True(MatrixVectorProduct(new double[,] {{}}, new double[] {}) != null);			
		}
		[Fact]
		public void MatrixVectorProduct_Evaluates_Input2()
		{
			//fill this.
			Assert.True(MatrixVectorProduct(new double[,] {{}}, new double[] {}) != null);			
		}
		[Fact]
		public void MatrixVectorProduct_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>( () => MatrixVectorProduct(new double[,] {{}}, new double[] {}));
		}
		[Fact]
		public void Add_Evaluates_Input1()
		{
			//fill this.
			Assert.True(Add(new double[] {}, new double[] {}) != null);			
		}
		[Fact]
		public void Add_Evaluates_Input2()
		{
			//fill this.
			Assert.True(Add(new double[] {}, new double[] {}) != null);			
		}
		[Fact]
		public void Add_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => Add(new double[] {}, new double[] {}));
		}
		[Fact]
		public void GeneralCNN_Evaluates_Input1()
		{
			//fill this.
			Assert.True(GeneralCNN(new double[,,] {{{}}}) != null);			
		}
		[Fact]
		public void GeneralCNN_Evaluates_Input2()
		{
			//fill this.		
			Assert.True(GeneralCNN(new double[,,] {{{}}}) != null);		
		}
		[Fact]
		public void GeneralCNN_Throws_Exception()
		{
			//fill this.
			Assert.Throws<Exception>(() => GeneralCNN(new double[,,] {{{}}}));
		}
    }
}
