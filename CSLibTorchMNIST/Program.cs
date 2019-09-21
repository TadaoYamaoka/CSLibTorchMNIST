using System;
using System.IO;
using System.Runtime.InteropServices;
using MathNet.Numerics;

namespace CSLibTorchMNIST
{
    class Program
    {
        [DllImport("mnist")]
        extern static void init();

        [DllImport("mnist")]
        unsafe extern static void train(float* dataset, long* targetset, int dataset_size);

        [DllImport("mnist")]
        unsafe extern static void test(float* dataset, long* targetset, int dataset_size);

        unsafe static void Main(string[] args)
        {
            init();

            // load mnist
            var imagesTrain = LoadImages(args[0]);
            var labelsTrain = LoadLabels(args[1]);
            var imagesTest = LoadImages(args[2]);
            var labelsTest = LoadLabels(args[3]);

            var imagesShuffled = new float[imagesTrain.GetLength(0), 28 * 28];
            var labelsShuffled = new long[labelsTrain.GetLength(0)];
            for (int epoch = 0; epoch < 10; ++epoch)
            {
                // shuffle
                var idxs = Combinatorics.GeneratePermutation(imagesTrain.GetLength(0));
                int i = 0;
                foreach (int idx in idxs)
                {
                    Array.Copy(imagesTrain, idx * 28 * 28, imagesShuffled, i * 28 * 28, 28 * 28);
                    labelsShuffled[i] = labelsTrain[idx];
                    ++i;
                }

                fixed(float* pImagesTrain = &imagesShuffled[0, 0])
                fixed(long* pLabelsTrain = &labelsShuffled[0])
                fixed(float* pImagesTest = &imagesTest[0, 0])
                fixed(long* pLabelsTest = &labelsTest[0])
                {
                    train(pImagesTrain, pLabelsTrain, imagesTrain.GetLength(0));
                    test(pImagesTest, pLabelsTest, imagesTest.GetLength(0));
                }
            }
        }

        static private float[,] LoadImages(string path)
        {
            using (BinaryReader reader = new BinaryReader(File.OpenRead(path)))
            {
                if (ReadInt32(reader) != 0x00000803)
                {
                    throw new FormatException();
                }

                var count = ReadInt32(reader);
                if (ReadInt32(reader) != 28)
                {
                    throw new FormatException();
                }
                if (ReadInt32(reader) != 28)
                {
                    throw new FormatException();
                }

                float[,] data = new float[count, 28 * 28];
                for (int i = 0; i < count; ++i)
                {
                    for (int j = 0; j < 28 * 28; ++j)
                    {
                        var v = reader.ReadByte();
                        // normalize
                        data[i, j] = ((v / 255f) -0.1307f) / 0.3081f;
                    }
                }

                return data;
            }
        }

        static private Int64[] LoadLabels(string path)
        {
            using (BinaryReader reader = new BinaryReader(File.OpenRead(path)))
            {
                if (ReadInt32(reader) != 0x00000801)
                {
                    throw new FormatException();
                }

                var count = ReadInt32(reader);

                Int64[] labels = new Int64[count];
                for (int i = 0; i < count; ++i)
                {
                    var v = reader.ReadByte();
                    labels[i] = v;
                }

                return labels;
            }
        }

        static private int ReadInt32(BinaryReader reader)
        {
            var v = reader.ReadBytes(4);
            Array.Reverse(v);
            return BitConverter.ToInt32(v);
        }
    }
}
