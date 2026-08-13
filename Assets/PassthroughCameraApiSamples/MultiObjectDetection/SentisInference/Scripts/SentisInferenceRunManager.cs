// Copyright (c) Meta Platforms, Inc. and affiliates.

using System.Collections;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;
using System.IO;

using Meta.XR;
using Meta.XR.Samples;
using Unity.Collections;
using Unity.InferenceEngine;
using UnityEngine;

namespace PassthroughCameraSamples.MultiObjectDetection
{
    [MetaCodeSample("PassthroughCameraApiSamples-MultiObjectDetection")]
    public class SentisInferenceRunManager : MonoBehaviour
    {
        [SerializeField] private PassthroughCameraAccess m_cameraAccess;
        [SerializeField] private DetectionUiMenuManager m_uiMenuManager;
        [SerializeField] private DetectionManager m_detectionManager;

        [Header("Sentis Model config")]
        [SerializeField] private BackendType m_backend = BackendType.CPU;
        [SerializeField] private ModelAsset m_sentisModel;
        [SerializeField] private TextAsset m_labelsAsset;
        [SerializeField, Range(0, 1)] private float m_iouThreshold = 0.6f;
        [SerializeField, Range(0, 1)] private float m_scoreThreshold = 0.23f;

        [Header("UI display references")]
        [SerializeField] private SentisInferenceUiManager m_uiInference;

        [Header("[Editor Only] Convert to Sentis")]
        public ModelAsset OnnxModel;
        [Space(40)]

        private Worker m_engine;
        private Vector2Int m_inputSize;

        // Detection sekarang menyimpan:
        // classId + confidence + boundingBox
       private readonly List<(int classId, Vector4 boundingBox)> m_detections =
            new List<(int classId, Vector4 boundingBox)>();

        private readonly List<float> m_detectionConfidences =
            new List<float>();

        // ============================================================
        // CSV LOGGING
        // ============================================================
        private StringBuilder m_logBuffer = new StringBuilder();
        private string m_logFilePath;

        private int m_inferenceId = 0;
        private int m_inferenceCounter = 0;

        // Tulis buffer ke disk setiap 30 inference.
        private const int FLUSH_EVERY_N_INFERENCES = 30;

        private void Awake()
        {
            var model = ModelLoader.Load(m_sentisModel);
            var inputShape = model.inputs[0].shape;
            m_inputSize = new Vector2Int(inputShape.Get(2), inputShape.Get(3));
            m_engine = new Worker(model, m_backend);

            // Simpan CSV di persistentDataPath Quest 3.
            m_logFilePath = Path.Combine(
                Application.persistentDataPath,
                "Inference_Log.csv"
            );

            // Buat file dan header jika belum ada.
            if (!File.Exists(m_logFilePath))
            {
                string header =
                    "Inference_ID,Timestamp_Sec,InferenceTime_Ms,ObjectName,Confidence,Box_X1,Box_Y1,Box_X2,Box_Y2\n";

                File.WriteAllText(m_logFilePath, header);
            }

            Debug.Log($"[SICS Data Log] CSV path: {m_logFilePath}");
        }

        /// <summary>
        /// Menambahkan satu hasil detection ke buffer CSV.
        /// Satu object = satu baris CSV.
        /// </summary>
        private void RecordInferenceLog(
            int inferenceId,
            float timestampSec,
            float inferenceTimeMs,
            string label,
            float confidence,
            Vector4 box)
        {
            string line =
                $"{inferenceId}," +
                $"{timestampSec:F2}," +
                $"{inferenceTimeMs:F2}," +
                $"{label}," +
                $"{confidence:F4}," +
                $"{box.x:F1}," +
                $"{box.y:F1}," +
                $"{box.z:F1}," +
                $"{box.w:F1}\n";

            m_logBuffer.Append(line);
        }

        /// <summary>
        /// Menambahkan satu inference yang tidak menghasilkan detection.
        /// </summary>
        private void RecordNoDetectionLog(
            int inferenceId,
            float timestampSec,
            float inferenceTimeMs)
        {
            string line =
                $"{inferenceId}," +
                $"{timestampSec:F2}," +
                $"{inferenceTimeMs:F2}," +
                "NONE,0,0,0,0,0\n";

            m_logBuffer.Append(line);
        }

        /// <summary>
        /// Menulis buffer RAM ke file CSV.
        /// </summary>
        private void FlushLogToFile()
        {
            if (m_logBuffer.Length == 0)
                return;

            try
            {
                File.AppendAllText(
                    m_logFilePath,
                    m_logBuffer.ToString()
                );

                m_logBuffer.Clear();
                m_inferenceCounter = 0;

                Debug.Log(
                    $"[LOG SUCCESS] Log inferensi berhasil ditulis ke: {m_logFilePath}"
                );
            }
            catch (System.Exception e)
            {
                Debug.LogError(
                    $"[LOG ERROR] Gagal menulis log: {e.Message}"
                );
            }
        }

        /// <summary>
        /// Mengambil nama class dari m_labelsAsset.
        /// Format labels diasumsikan satu class per baris:
        /// 0 = bottle
        /// 1 = eraser
        /// 2 = pen
        /// </summary>
        private string GetClassName(int classId)
        {
            if (m_labelsAsset == null)
                return $"class_{classId}";

            string[] labels = m_labelsAsset.text
                .Split(new[] { '\r', '\n' }, System.StringSplitOptions.RemoveEmptyEntries);

            if (classId >= 0 && classId < labels.Length)
            {
                string label = labels[classId].Trim();

                // Jika file label ternyata berformat "0 bottle",
                // ambil bagian setelah nomor class.
                int separatorIndex = label.IndexOfAny(new[] { ' ', '\t' });

                if (separatorIndex > 0)
                {
                    string possibleId = label.Substring(0, separatorIndex).Trim();

                    if (int.TryParse(possibleId, out int parsedId) &&
                        parsedId == classId)
                    {
                        label = label.Substring(separatorIndex).Trim();
                    }
                }

                return label;
            }

            return $"class_{classId}";
        }

        private IEnumerator Start()
        {
            m_uiInference.SetLabels(m_labelsAsset);

            while (true)
            {
                while (m_uiMenuManager.IsPaused)
                {
                    yield return null;
                }

                yield return RunInference();
            }
        }

        // ============================================================
        // QUEST 3 LIFECYCLE / SAVE LOG
        // ============================================================

        private void OnApplicationPause(bool pauseStatus)
        {
            if (pauseStatus)
            {
                FlushLogToFile();
            }
        }

        private void OnDisable()
        {
            FlushLogToFile();
        }

        private void OnDestroy()
        {
            // Simpan sisa log sebelum object dihancurkan.
            FlushLogToFile();

            if (m_engine != null)
            {
                m_engine.PeekOutput(0)?.CompleteAllPendingOperations();
                m_engine.PeekOutput(1)?.CompleteAllPendingOperations();
                m_engine.PeekOutput(2)?.CompleteAllPendingOperations();
                m_engine.Dispose();
                m_engine = null;
            }
        }

        private void OnApplicationQuit()
        {
            FlushLogToFile();
        }

        internal static void PreloadModel(ModelAsset modelAsset)
        {
            var model = ModelLoader.Load(modelAsset);
            var inputShape = model.inputs[0].shape;

            using var worker = new Worker(model, BackendType.CPU);

            Texture tempTexture =
                new Texture2D(2, 2, TextureFormat.RGBA32, false);

            var textureTransform =
                new TextureTransform().SetDimensions(
                    tempTexture.width,
                    tempTexture.height,
                    3
                );

            using var input =
                new Tensor<float>(
                    new TensorShape(
                        1,
                        3,
                        inputShape.Get(2),
                        inputShape.Get(3)
                    )
                );

            TextureConverter.ToTensor(
                tempTexture,
                input,
                textureTransform
            );

            worker.Schedule(input);

            worker.PeekOutput(0).CompleteAllPendingOperations();
            worker.PeekOutput(1).CompleteAllPendingOperations();
            worker.PeekOutput(2).CompleteAllPendingOperations();

            Destroy(tempTexture);
        }

        private IEnumerator RunInference()
        {
            if (!m_cameraAccess.IsPlaying)
            {
                yield break;
            }

            [DllImport("OVRPlugin", CallingConvention = CallingConvention.Cdecl)]
            static extern OVRPlugin.Result ovrp_GetNodePoseStateAtTime(
                double time,
                OVRPlugin.Node nodeId,
                out OVRPlugin.PoseStatef nodePoseState
            );

            if (!ovrp_GetNodePoseStateAtTime(
                    OVRPlugin.GetTimeInSeconds(),
                    OVRPlugin.Node.Head,
                    out _).IsSuccess())
            {
                Debug.Log(
                    "ovrp_GetNodePoseStateAtTime failed, which means " +
                    "'m_cameraAccess.GetCameraPose()' is not reliable, skipping."
                );

                yield break;
            }

            var cachedCameraPose = m_cameraAccess.GetCameraPose();

            Texture targetTexture = m_cameraAccess.GetTexture();

            var textureTransform =
                new TextureTransform().SetDimensions(
                    targetTexture.width,
                    targetTexture.height,
                    3
                );

            using var input =
                new Tensor<float>(
                    new TensorShape(
                        1,
                        3,
                        m_inputSize.x,
                        m_inputSize.y
                    )
                );

            TextureConverter.ToTensor(
                targetTexture,
                input,
                textureTransform
            );

            // ========================================================
            // MULAI PENGUKURAN INFERENCE
            // ========================================================
            System.Diagnostics.Stopwatch stopwatch =
                System.Diagnostics.Stopwatch.StartNew();

            m_engine.Schedule(input);

            var boxesAwaiter =
                (m_engine.PeekOutput(0) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();

            while (!boxesAwaiter.IsCompleted)
            {
                yield return null;
            }

            using var boxes = boxesAwaiter.GetResult();

            if (boxes.shape[0] == 0)
            {
                yield break;
            }

            var classIDsAwaiter =
                (m_engine.PeekOutput(1) as Tensor<int>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();

            while (!classIDsAwaiter.IsCompleted)
            {
                yield return null;
            }

            using var classIDs = classIDsAwaiter.GetResult();

            if (classIDs.shape[0] == 0)
            {
                Debug.LogError("classIDs.shape[0] == 0");
                yield break;
            }

            var scoresAwaiter =
                (m_engine.PeekOutput(2) as Tensor<float>)
                .ReadbackAndCloneAsync()
                .GetAwaiter();

            while (!scoresAwaiter.IsCompleted)
            {
                yield return null;
            }

            using var scores = scoresAwaiter.GetResult();

            if (scores.shape[0] == 0)
            {
                Debug.LogError("scores.shape[0] == 0");
                yield break;
            }

            stopwatch.Stop();

            float inferenceTimeMs =
                (float)stopwatch.Elapsed.TotalMilliseconds;

            // Timestamp = waktu sejak aplikasi Unity mulai.
            float timestampSec = Time.time;

            // Satu ID untuk satu siklus inference.
            m_inferenceId++;
            m_inferenceCounter++;

            // ========================================================
            // UI INFERENCE TIME
            // ========================================================
            if (m_uiInference != null)
            {
                m_uiInference.UpdateInferenceTime(
                    stopwatch.ElapsedMilliseconds
                );
            }

            // ========================================================
            // NMS
            // ========================================================
            NonMaxSuppression(
                m_detections,
                m_detectionConfidences,
                boxes,
                classIDs,
                scores,
                m_iouThreshold,
                m_scoreThreshold
            );

            // ========================================================
            // CSV LOGGING
            // ========================================================
            if (m_detections.Count > 0)
            {
                for (int i = 0; i < m_detections.Count; i++)
                {
                    var detection = m_detections[i];

                    string objectName =
                        GetClassName(detection.classId);

                    float confidence =
                        m_detectionConfidences[i];

                    RecordInferenceLog(
                        m_inferenceId,
                        timestampSec,
                        inferenceTimeMs,
                        objectName,
                        confidence,
                        detection.boundingBox
                    );
                }
            }
            else
            {
                // Tetap catat inference walaupun tidak ada object.
                RecordNoDetectionLog(
                    m_inferenceId,
                    timestampSec,
                    inferenceTimeMs
                );
            }

            // Flush berdasarkan jumlah inference,
            // bukan jumlah object detection.
            if (m_inferenceCounter >= FLUSH_EVERY_N_INFERENCES)
            {
                FlushLogToFile();
            }

            // ========================================================
            // CHECK SPATIAL ANCHOR
            // ========================================================
            if (!m_cameraAccess.IsPlaying ||
                m_detectionManager.m_spatialAnchor == null ||
                !m_detectionManager.m_spatialAnchor.IsTracked)
            {
                yield break;
            }

            // ========================================================
            // UPDATE UI BOXES
            // ========================================================
            m_uiInference.DrawUIBoxes(
                m_detections,
                m_inputSize,
                cachedCameraPose
            );
        }

        private static void NonMaxSuppression(
            List<(int classId, Vector4 boundingBox)> outDetections,
            List<float> outConfidences,
            Tensor<float> boxes,
            Tensor<int> classIDs,
            Tensor<float> scores,
            float iouThreshold,
            float scoreThreshold)
        {
            outDetections.Clear();

            // Filter berdasarkan score threshold.
            List<int> filteredIndices = new List<int>();

            NativeArray<float>.ReadOnly scoresArray =
                scores.AsReadOnlyNativeArray();

            for (int i = 0; i < scoresArray.Length; i++)
            {
                if (scoresArray[i] >= scoreThreshold)
                {
                    filteredIndices.Add(i);
                }
            }

            if (filteredIndices.Count == 0)
            {
                return;
            }

            // Urutkan confidence tertinggi terlebih dahulu.
            filteredIndices.Sort(
                (a, b) => scoresArray[b].CompareTo(scoresArray[a])
            );

            // Apply NMS.
            bool[] suppressed =
                new bool[filteredIndices.Count];

            for (int i = 0; i < filteredIndices.Count; i++)
            {
                if (suppressed[i])
                    continue;

                int idx = filteredIndices[i];

                Vector4 currentBox = GetBox(idx);
                float currentConfidence = scoresArray[idx];
                int currentClassId = classIDs[idx];

                outDetections.Add(
                    (
                        currentClassId,
                        currentBox
                    )
                );

                outConfidences.Add(currentConfidence);

                // Suppress overlapping boxes.
                for (int j = i + 1; j < filteredIndices.Count; j++)
                {
                    if (suppressed[j])
                        continue;

                    int jdx = filteredIndices[j];

                    float iou =
                        CalculateIoU(
                            currentBox,
                            GetBox(jdx)
                        );

                    if (iou > iouThreshold)
                    {
                        suppressed[j] = true;
                    }
                }
            }

            Vector4 GetBox(int i)
            {
                return new Vector4(
                    boxes[i, 0],
                    boxes[i, 1],
                    boxes[i, 2],
                    boxes[i, 3]
                );
            }
        }

        internal static float CalculateIoU(
            Vector4 boxA,
            Vector4 boxB)
        {
            // Format:
            // (topLeftX, topLeftY, bottomRightX, bottomRightY)

            float x1 = Mathf.Max(boxA.x, boxB.x);
            float y1 = Mathf.Max(boxA.y, boxB.y);
            float x2 = Mathf.Min(boxA.z, boxB.z);
            float y2 = Mathf.Min(boxA.w, boxB.w);

            float intersectionWidth =
                Mathf.Max(0, x2 - x1);

            float intersectionHeight =
                Mathf.Max(0, y2 - y1);

            float intersectionArea =
                intersectionWidth * intersectionHeight;

            float boxAArea =
                (boxA.z - boxA.x) *
                (boxA.w - boxA.y);

            float boxBArea =
                (boxB.z - boxB.x) *
                (boxB.w - boxB.y);

            float unionArea =
                boxAArea +
                boxBArea -
                intersectionArea;

            if (unionArea == 0)
                return 0;

            return intersectionArea / unionArea;
        }
    }
}
