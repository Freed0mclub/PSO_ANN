using System;
using System.Collections.Generic;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using PSO_ANN.ANN;
using PSO_ANN.PSO;

namespace PSO_ANN
{
    internal class Program
    {
        static void Main(string[] args)
        {
            // 1) Locate the CSV in the project DATA folder
            string exeDir = AppDomain.CurrentDomain.BaseDirectory;
            // exeDir ~ ...\PSO_ANN\bin\Debug\net8.0\
            string projectDir = Path.GetFullPath(Path.Combine(exeDir, "..", "..", ".."));
            string csvPath = Path.Combine(projectDir, "DATA", "housing.csv");

            if (!File.Exists(csvPath))
                throw new FileNotFoundException($"Could not find housing.csv at: {csvPath}");

            // 2) Load and parse the CSV
            var allData = File.ReadAllLines(csvPath)
                              .Where(line => !string.IsNullOrWhiteSpace(line))
                              .Select(line =>
                              {
                                  var parts = line.Split(',');
                                  int fCount = parts.Length - 1;
                                  var inputs = new double[fCount];
                                  for (int i = 0; i < fCount; i++)
                                      inputs[i] = double.Parse(parts[i], CultureInfo.InvariantCulture);
                                  var target = double.Parse(parts[fCount], CultureInfo.InvariantCulture);
                                  return (
                                            inputs: inputs,
                                            targets: new double[] { target }
                                            );
                              })
                              .ToList();

            // 3) Normalize features (min–max)
            int featureCount = allData[0].inputs.Length;
            var mins = Enumerable.Range(0, featureCount)
                                 .Select(i => allData.Min(d => d.inputs[i]))
                                 .ToArray();
            var maxs = Enumerable.Range(0, featureCount)
                                 .Select(i => allData.Max(d => d.inputs[i]))
                                 .ToArray();

            foreach (var (inputs, _) in allData)
                for (int i = 0; i < featureCount; i++)
                    inputs[i] = (inputs[i] - mins[i]) / (maxs[i] - mins[i]);

            // 4) Train/validation split
            var rng = new Random(42);
            var shuffled = allData.OrderBy(_ => rng.Next()).ToList();
            int splitIdx = (int)(shuffled.Count * 0.8);
            var trainSet = shuffled.Take(splitIdx).ToList();
            var valSet = shuffled.Skip(splitIdx).ToList();

            // 5) Build ANN + PSO
            var topology = new[] { featureCount, 10, 1 };
            var ann = new NeuralNetwork(topology);
            int dimensions = ann.GetWeightsCount();
            var swarm = new Swarm(particleCount: 50, dimensions: dimensions);
            int iterations = 1000;

            // 6) Define fitness = train MSE
            Func<double[], double> fitness = weights =>
            {
                ann.SetWeights(weights);
                double mse = 0;
                foreach (var (inputs, target) in trainSet)
                {
                    var o = ann.Forward(inputs);
                    mse += Math.Pow(o[0] - target[0], 2);
                }
                return mse / trainSet.Count;
            };

            // 7) Run optimization
            for (int i = 1; i <= iterations; i++)
            {
                swarm.UpdateParticles(fitness);
                if (i % 100 == 0)
                    Console.WriteLine($"Iteration {i}/{iterations}, Best Train MSE: {swarm.GlobalBestFitness:F4}");
            }

            // 8) Evaluate on validation set
            ann.SetWeights(swarm.GlobalBestPosition);
            double valMse = valSet.Sum(d =>
            {
                var o = ann.Forward(d.inputs);
                return Math.Pow(o[0] - d.targets[0], 2);
            }) / valSet.Count;

            Console.WriteLine($"\nValidation MSE: {valMse:F4}");
        }
    }
}
