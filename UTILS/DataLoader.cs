using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using PSO_ANN.ANN;

namespace PSO_ANN.UTILS
{
    public static class DataLoader
    {
        /// <summary>
        /// Loads a CSV where the last column is the target value and the preceding columns are features.
        /// </summary>
        public static List<(double[] inputs, double[] targets)> Load(string relativePath)
        {
            var records = new List<(double[], double[])>();
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string fullPath = Path.Combine(baseDir, relativePath);

            if (!File.Exists(fullPath))
                throw new FileNotFoundException($"Dataset file not found: {fullPath}");

            using var reader = new StreamReader(fullPath);
            string line;

            while ((line = reader.ReadLine()) != null)
            {
                if (string.IsNullOrWhiteSpace(line))
                    continue;

                var parts = line.Split(',');
                int featureCount = parts.Length - 1;
                var inputs = new double[featureCount];

                for (int i = 0; i < featureCount; i++)
                {
                    if (!double.TryParse(parts[i], NumberStyles.Float, CultureInfo.InvariantCulture, out double val))
                        throw new FormatException($"Invalid feature value '{parts[i]}' at line: {line}");

                    inputs[i] = val;
                }

                if (!double.TryParse(parts[featureCount], NumberStyles.Float, CultureInfo.InvariantCulture, out double target))
                    throw new FormatException($"Invalid target value '{parts[featureCount]}' at line: {line}");

                records.Add((inputs, new double[] { target }));
            }

            return records;
        }
    }
}
