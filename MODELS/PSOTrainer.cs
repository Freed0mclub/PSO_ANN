using PSO_ANN.ANN;
using PSO_ANN.PSO;
using PSO_ANN.UTILS;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace PSO_ANN.MODELS
{
    public static class PSOTrainer
    {

        /// Runs PSO-only training on a normalized dataset.
        /// <param name="csvPath">Path to the space-delimited CSV file.</param>
        /// <param name="hiddenNeurons">Number of neurons in the hidden layer.</param>
        /// <param name="particleCount">Number of PSO particles.</param>
        /// <param name="iterations">Number of PSO iterations.</param>
        /// Tuple of (bestWeights, trainMSE, validationMSE).

        public static (double[] bestWeights, double trainMse, double valMse) Run(
            string csvPath,
            int hiddenNeurons = 10,
            int particleCount = 50,
            int iterations = 1000)
        {
            // 1) Load & normalize data
            var data = DataLoader.LoadAndNormalize(csvPath);
            int nFeatures = data[0].inputs.Length;

            // 2) Split into train/validation (80/20)
            var rng = new Random(42);
            var shuffled = data.OrderBy(_ => rng.Next()).ToList();
            int splitIdx = (int)(0.8 * shuffled.Count);
            var trainSet = shuffled.Take(splitIdx).ToList();
            var valSet = shuffled.Skip(splitIdx).ToList();

            // 3) Initialize ANN and PSO
            var topology = new[] { nFeatures, hiddenNeurons, 1 };
            var ann = new NeuralNetwork(topology);
            int dims = ann.GetWeightsCount();
            var swarm = new Swarm(particleCount, dims);

            // 4) Define fitness function (MSE on train)
            Func<double[], double> fitness = weights =>
            {
                ann.SetWeights(weights);
                double mse = 0;
                foreach (var (inputs, targets) in trainSet)
                {
                    var output = ann.Forward(inputs);
                    mse += Math.Pow(output[0] - targets[0], 2);
                }
                return mse / trainSet.Count;
            };

            // 5) Run PSO optimization
            for (int i = 1; i <= iterations; i++)
                swarm.UpdateParticles(fitness);

            // 6) Evaluate on validation set
            ann.SetWeights(swarm.GlobalBestPosition);
            double valMse = valSet.Sum(d =>
            {
                var o = ann.Forward(d.inputs);
                return Math.Pow(o[0] - d.targets[0], 2);
            }) / valSet.Count;

            // 7) Return best weights and errors
            return (swarm.GlobalBestPosition, swarm.GlobalBestFitness, valMse);
        }
    }
}
