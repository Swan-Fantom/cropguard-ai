const classMetrics = [
  { name: "Common Rust", support: 109, precision: 1.0, recall: 0.97, f1: 0.985 },
  { name: "Gray Leaf Spot", support: 25, precision: 0.89, recall: 1.0, f1: 0.942 },
  { name: "Healthy", support: 104, precision: 0.99, recall: 1.0, f1: 0.995 },
  { name: "Northern Leaf Blight", support: 100, precision: 1.0, recall: 1.0, f1: 1.0 },
];

const confusionMatrix = [
  ["Actual / Predicted", "Rust", "Gray spot", "Healthy", "Blight"],
  ["Rust", 106, 3, 0, 0],
  ["Gray spot", 0, 25, 0, 0],
  ["Healthy", 0, 0, 104, 0],
  ["Blight", 0, 0, 0, 100],
];

function percentage(value) {
  return `${Math.round(value * 1000) / 10}%`;
}

export default function EvaluationPage() {
  return (
    <div>
      <h1 className="mb-1 text-2xl font-semibold">Model evaluation</h1>
      <p className="mb-6 text-sm text-gray-500">
        Performance measured on 338 held-out field images that were not used to train the model.
      </p>

      <div className="grid gap-4 sm:grid-cols-3">
        {[
          ["Accuracy", "98.8%"],
          ["Macro-F1", "98.1%"],
          ["Test images", "338"],
        ].map(([label, value]) => (
          <div key={label} className="rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
            <div className="text-sm text-gray-500">{label}</div>
            <div className="mt-1 text-2xl font-semibold text-brand-700">{value}</div>
          </div>
        ))}
      </div>

      <section className="mt-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold">Per-class test scores</h2>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-left text-sm">
            <thead className="border-b border-gray-200 text-gray-500">
              <tr>
                <th className="px-3 py-2 font-medium">Condition</th>
                <th className="px-3 py-2 font-medium">Precision</th>
                <th className="px-3 py-2 font-medium">Recall</th>
                <th className="px-3 py-2 font-medium">F1</th>
                <th className="px-3 py-2 font-medium">Images</th>
              </tr>
            </thead>
            <tbody>
              {classMetrics.map((metric) => (
                <tr key={metric.name} className="border-b border-gray-100 last:border-0">
                  <td className="px-3 py-2 font-medium text-gray-800">{metric.name}</td>
                  <td className="px-3 py-2">{percentage(metric.precision)}</td>
                  <td className="px-3 py-2">{percentage(metric.recall)}</td>
                  <td className="px-3 py-2">{percentage(metric.f1)}</td>
                  <td className="px-3 py-2">{metric.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-6 rounded-xl border border-gray-200 bg-white p-5 shadow-sm">
        <h2 className="text-lg font-semibold">Confusion matrix</h2>
        <p className="mt-1 text-sm text-gray-500">
          Rows are the actual condition and columns are the model prediction.
        </p>
        <div className="mt-4 overflow-x-auto">
          <table className="min-w-full text-center text-sm">
            <tbody>
              {confusionMatrix.map((row, rowIndex) => (
                <tr key={row[0]}>
                  {row.map((value, columnIndex) => (
                    <td
                      key={`${row[0]}-${columnIndex}`}
                      className={`border border-gray-200 px-3 py-2 ${
                        rowIndex === 0 || columnIndex === 0
                          ? "bg-gray-50 font-medium text-gray-700"
                          : "text-gray-800"
                      }`}
                    >
                      {value}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-6 rounded-xl border border-amber-200 bg-amber-50 p-5 text-sm text-amber-900">
        <h2 className="font-semibold">Interpretation and limitations</h2>
        <ul className="mt-2 list-disc space-y-1 pl-5">
          <li>
            Results come from a held-out test split, not the training images, but the test set is
            still relatively small and may not represent every farm or season.
          </li>
          <li>
            Lighting, shadows, blur, camera quality, leaf angle, background clutter, and multiple
            leaves in one photo can reduce reliability.
          </li>
          <li>
            Gray Leaf Spot has only 25 test images, so its precision and recall are more uncertain
            than the headline accuracy.
          </li>
        </ul>
      </section>

      <p className="mt-6 rounded-lg border border-brand-200 bg-brand-50 px-4 py-3 text-sm text-brand-900">
        CropGuard supports early screening and prioritises cases for inspection. It does not
        replace agricultural expert advice or an on-site diagnosis. Low-confidence results require
        expert inspection.
      </p>
    </div>
  );
}
