"use client";

import { useEffect, useState } from "react";
import { EmptyState, ErrorState, MetricCard, SectionHeading, Skeleton } from "../components/dashboard";
import { api } from "../lib/api";
import { formatInteger, formatMetric, formatPercent } from "../lib/research-formatters";
import type { ResearchFeaturesResponse, ResearchModelResponse, ResearchPredictionsResponse } from "../types";
import { ResearchChart } from "./research-chart";

function StatusLine({ label, available }: { label: string; available: boolean }) {
  return <div className="research-status-line"><span className={available ? "status-dot status-dot-live" : "status-dot status-dot-error"} /> <span>{label}</span><strong>{available ? "Available" : "Unavailable"}</strong></div>;
}

export default function ResearchPage() {
  const [model, setModel] = useState<ResearchModelResponse | null>(null);
  const [predictions, setPredictions] = useState<ResearchPredictionsResponse | null>(null);
  const [features, setFeatures] = useState<ResearchFeaturesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [modelFailed, setModelFailed] = useState(false);
  const [predictionsFailed, setPredictionsFailed] = useState(false);
  const [featuresFailed, setFeaturesFailed] = useState(false);

  useEffect(() => {
    Promise.allSettled([api.researchModel(), api.researchPredictions(100), api.researchFeatures()]).then(([modelResult, predictionResult, featureResult]) => {
      if (modelResult.status === "fulfilled") setModel(modelResult.value); else setModelFailed(true);
      if (predictionResult.status === "fulfilled") setPredictions(predictionResult.value); else setPredictionsFailed(true);
      if (featureResult.status === "fulfilled") setFeatures(featureResult.value); else setFeaturesFailed(true);
      setLoading(false);
    });
  }, []);

  const actualColumn = predictions?.available_columns.find((column) => ["actual_return_5d", "actual", "target"].includes(column.toLowerCase()));
  const predictedColumn = predictions?.available_columns.find((column) => ["predicted_return_5d", "predicted", "prediction"].includes(column.toLowerCase()));
  const hasPredictionChart = Boolean(actualColumn && predictedColumn && predictions?.observations.length);

  return (
    <>
      <div className="content-header"><div><span className="eyebrow">Model intelligence</span><h1>Research</h1><p>Model diagnostics, predictive performance, and research analytics.</p></div></div>
      {modelFailed && predictionsFailed && featuresFailed ? <ErrorState message="Research data unavailable. Check that the model artifact and OOS prediction files are accessible to the API." /> : null}

      <section className="research-status-grid" aria-label="Research artifact status">
        <div className="panel research-status-panel"><div className="panel-header"><SectionHeading eyebrow="Artifact status" title="Research inputs" /></div><div className="research-status-list"><StatusLine label="Model artifact" available={!modelFailed && Boolean(model)} /><StatusLine label="OOS predictions" available={!predictionsFailed && Boolean(predictions)} /><StatusLine label="Feature metadata" available={!featuresFailed && Boolean(features)} /></div></div>
        <div className="research-disclosure"><span className="eyebrow">Evaluation context</span><p>Model metrics describe historical out-of-sample evaluation and are not a guarantee of future performance.</p></div>
      </section>

      <section aria-label="Model summary"><div className="research-section-heading"><SectionHeading eyebrow="Model specification" title="XGBoost return predictor" /></div><div className="metric-grid research-metric-grid">{loading ? <><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /><Skeleton className="skeleton-card" /></> : <><MetricCard label="Model" value={model?.model.name ?? "N/A"} detail={model?.model.type ?? "Type unavailable"} /><MetricCard label="Target" value={model?.model.target ?? "N/A"} detail="Saved artifact target" /><MetricCard label="Forecast horizon" value={model?.model.horizon_days == null ? "N/A" : `${model.model.horizon_days} days`} detail="Forward return horizon" /><MetricCard label="Model features" value={formatInteger(model?.feature_count)} detail="Saved feature columns" /></>}</div></section>

      <section className="panel research-panel evaluation-panel" aria-label="Out-of-sample evaluation metrics"><div className="panel-header"><SectionHeading eyebrow="Test evaluation" title="Out-of-sample performance" action={<span className="as-of">Computed from OOS predictions</span>} /></div>{loading ? <div className="research-metric-list"><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></div> : model ? <div className="research-metric-list"><div><span className="eyebrow">Mean absolute error</span><strong>{formatMetric(model.metrics.mae, 4)}</strong></div><div><span className="eyebrow">Root mean squared error</span><strong>{formatMetric(model.metrics.rmse, 4)}</strong></div><div><span className="eyebrow">Directional accuracy</span><strong>{formatPercent(model.metrics.directional_accuracy)}</strong></div><div><span className="eyebrow">Prediction correlation</span><strong>{formatMetric(model.metrics.prediction_correlation, 3)}</strong></div></div> : <EmptyState title="Evaluation unavailable" message="The model artifact or OOS observations could not be read." />}</section>

      <div className="research-lower-grid">
        <section className="panel research-panel" aria-label="Dataset and validation"><div className="panel-header"><SectionHeading eyebrow="Dataset validation" title="Evaluation sample" /></div><div className="dataset-list"><div><span>Total rows</span><strong>{formatInteger(model?.dataset.rows)}</strong></div><div><span>Target rows</span><strong>{formatInteger(model?.dataset.target_rows)}</strong></div><div><span>Training observations</span><strong>{formatInteger(model?.dataset.train_rows)}</strong></div><div><span>Validation observations</span><strong>{formatInteger(model?.dataset.validation_rows)}</strong></div><div><span>Test observations</span><strong>{formatInteger(model?.dataset.test_rows)}</strong></div><div><span>Split methodology</span><strong>Chronological</strong></div></div><p className="panel-note">Train and validation counts are not persisted in the saved artifacts and are shown as N/A when they cannot be verified.</p></section>
        <section className="panel research-panel" aria-label="Out-of-sample predictions"><div className="panel-header"><SectionHeading eyebrow="Prediction diagnostics" title="Actual vs predicted returns" action={predictions ? <span className="as-of">Latest {predictions.observations.length} observations</span> : null} /></div>{predictionsFailed ? <ErrorState message="OOS prediction data unavailable." /> : predictions && hasPredictionChart && actualColumn && predictedColumn ? <ResearchChart observations={predictions.observations} actualColumn={actualColumn} predictedColumn={predictedColumn} /> : <EmptyState title="Chart unavailable" message="The available OOS artifact does not contain both actual and predicted return columns." />}</section>
      </div>

      <section className="panel research-panel feature-panel" aria-label="Model features"><div className="panel-header"><SectionHeading eyebrow="Feature explorer" title="Saved model features" action={features ? <span className="as-of">{features.feature_count} features</span> : null} /></div>{featuresFailed ? <ErrorState message="Feature metadata unavailable from the model artifact." /> : features ? <div className="feature-list">{features.features.map((feature) => <span key={feature}>{feature}</span>)}</div> : <div className="feature-skeleton"><Skeleton className="skeleton-line" /><Skeleton className="skeleton-line" /></div>}</section>
    </>
  );
}
