using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace PSO_ANN.ANN
{
    public class NeuralNetwork
    {
        public Layer[] Layers { get; private set; }

        public NeuralNetwork(int[] topology)
        {
            Layers = new Layer[topology.Length - 1];

            for (int i = 0; i < Layers.Length; i++)
                Layers[i] = new Layer(topology[i], topology[i + 1]);
        }

        public double[] Forward(double[] inputs)
        {
            double[] output = inputs;

            foreach (var layer in Layers)
                output = layer.Compute(output);

            return output;
        }

        public void SetWeights(double[] weights)
        {
            int pos = 0;

            foreach (var layer in Layers)
                pos = layer.SetWeights(weights, pos);
        }

        public int GetWeightsCount()
        {
            int count = 0;

            foreach (var layer in Layers)
                count += layer.GetWeightsCount();

            return count;
        }
    }
}