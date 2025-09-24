export async function loadXGBoostModel(path, options = {}) {
  const json = await loadModelJson(path);
  return new XGBoostModel(json, options);
}

async function loadModelJson(path) {
  if (typeof window === 'undefined') {
    const { readFile } = await import('node:fs/promises');
    const url = new URL(path, import.meta.url);
    const buffer = await readFile(url, 'utf-8');
    return JSON.parse(buffer);
  }
  const response = await fetch(path, { cache: 'no-cache' });
  if (!response.ok) {
    throw new Error(`Failed to load model asset: ${path} (${response.status})`);
  }
  return await response.json();
}

class XGBoostModel {
  constructor(rawModel, options = {}) {
    const learner = rawModel.learner;
    const boosterModel = learner.gradient_booster.model;
    this.featureNames = learner.feature_names;
    const params = learner.learner_model_param;
    this.numClass = Number.parseInt(
      params.num_class ?? learner.objective?.softmax_multiclass_param?.num_class ?? '1',
      10
    );
    const baseScore = Number.parseFloat(params.base_score ?? '0');
    this.baseMargin = computeBaseMargin(baseScore);
    this.treeInfo = boosterModel.tree_info.map((value) => Number(value));
    this.trees = boosterModel.trees.map((tree) => normalizeTree(tree));
    this.ece = typeof options.ece === 'number' ? options.ece : null;
  }

  predict(featureMap = {}) {
    const rawScores = new Array(this.numClass).fill(this.baseMargin);
    for (let i = 0; i < this.trees.length; i += 1) {
      const tree = this.trees[i];
      const classIndex = this.treeInfo[i] ?? 0;
      rawScores[classIndex] += evaluateTree(tree, this.featureNames, featureMap);
    }
    const probs = softmax(rawScores);
    const predClass = probs.indexOf(Math.max(...probs));
    return {
      model: 'xgboost_classifier',
      raw_scores: rawScores,
      pred_class: predClass,
      probs,
      top2: findTopIndices(probs, 2),
      ece_bin: this.ece,
      explanation: []
    };
  }
}

function normalizeTree(tree) {
  return {
    left: tree.left_children.map((value) => Number(value)),
    right: tree.right_children.map((value) => Number(value)),
    splitIndex: tree.split_indices.map((value) => Number(value)),
    threshold: tree.split_conditions.map((value) => Number(value)),
    baseWeight: tree.base_weights.map((value) => Number(value)),
    defaultLeft: tree.default_left.map((value) => Number(value) === 1)
  };
}

function evaluateTree(tree, featureNames, featureMap) {
  let node = 0;
  while (true) {
    const left = tree.left[node];
    const right = tree.right[node];
    if (left === -1 && right === -1) {
      return tree.baseWeight[node];
    }
    const featureId = tree.splitIndex[node];
    const featureName = featureNames[featureId];
    const rawValue = featureMap[featureName];
    const numericValue = typeof rawValue === 'number' ? rawValue : Number(rawValue);
    const missing = rawValue === null || typeof rawValue === 'undefined' || Number.isNaN(numericValue);
    const goLeft = missing ? tree.defaultLeft[node] : numericValue < tree.threshold[node];
    node = goLeft ? left : right;
  }
}

function computeBaseMargin(baseScore) {
  if (Number.isFinite(baseScore) && baseScore > 0 && baseScore < 1) {
    return Math.log(baseScore / (1 - baseScore));
  }
  if (Number.isFinite(baseScore)) {
    return baseScore;
  }
  return 0;
}

function softmax(values) {
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - max));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

function findTopIndices(values, count) {
  return values
    .map((value, index) => ({ value, index }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count)
    .map((item) => item.index);
}
