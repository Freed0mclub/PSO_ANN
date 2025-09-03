using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace PSO_ANN.UTILS
{
	
	/// Minimal CSV logger for per-iteration metrics. Thread-safe for single-writer patterns
	/// (typical in trainers). Creates the directory if missing. UTF-8 without BOM.
	
	public sealed class CsvLogger : IDisposable
	{
		private readonly StreamWriter _writer;
		private readonly string[] _columns;
		private bool _headerWritten;

		public string Path { get; }

		public CsvLogger(string path, params string[] columns)
		{
			if (columns == null || columns.Length == 0)
				throw new ArgumentException("At least one column name is required.", nameof(columns));

			Path = System.IO.Path.GetFullPath(path);
			Directory.CreateDirectory(System.IO.Path.GetDirectoryName(Path)!);

			_writer = new StreamWriter(new FileStream(Path, FileMode.Create, FileAccess.Write, FileShare.Read))
			{
				NewLine = "\n"
			};
			_columns = columns;
		}

		public void WriteHeader()
		{
			if (_headerWritten) return;
			_writer.WriteLine(string.Join(",", _columns.Select(Escape)));
			_writer.Flush();
			_headerWritten = true;
		}

		
		/// Write a row of values in the same order as the columns.
		/// Values are formatted invariantly and CSV-escaped.
		
		public void WriteRow(params object[] values)
		{
			if (!_headerWritten) WriteHeader();
			if (values.Length != _columns.Length)
				throw new ArgumentException($"Expected {_columns.Length} values, got {values.Length}.");

			string[] cells = new string[values.Length];
			for (int i = 0; i < values.Length; i++)
			{
				var v = values[i];
				if (v is IFormattable f)
					cells[i] = Escape(f.ToString(null, CultureInfo.InvariantCulture) ?? string.Empty);
				else
					cells[i] = Escape(v?.ToString() ?? string.Empty);
			}
			_writer.WriteLine(string.Join(",", cells));
		}

		public void Flush() => _writer.Flush();

		private static string Escape(string s)
		{
			if (s.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0)
				return '"' + s.Replace("\"", "\"\"") + '"';
			return s;
		}

		public void Dispose()
		{
			_writer.Flush();
			_writer.Dispose();
		}
	}
}
