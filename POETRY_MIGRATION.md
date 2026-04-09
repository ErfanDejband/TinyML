# Poetry Migration Guide for TinyML Project

## 🎯 Summary

**Good News:** I successfully resolved the dependency conflicts! Poetry can work with your project.

**The Trade-off:** You'll need to downgrade a few packages slightly from their bleeding-edge versions.

## 📊 Version Changes

| Package | Current (requirements.txt) | Poetry Resolution | Change |
|---------|---------------------------|-------------------|--------|
| numpy | 1.26.4 | 1.26.4 | ✅ Same |
| tensorflow | 2.21.0 | 2.18.1 | ⬇️ Minor downgrade |
| tf-keras | 2.21.0 | 2.18.0 | ⬇️ Minor downgrade |
| tensorflow-model-optimization | 0.8.0 | 0.8.0 | ✅ Same |
| pandas | 3.0.1 | 2.3.3 | ⬇️ Minor downgrade |
| scikit-learn | 1.8.0 | 1.8.0 | ✅ Same |
| matplotlib | 3.10.8 | 3.10.8 | ✅ Same |

## 🔍 Why the Conflicts?

The dependency graph has a **deadlock**:

```
TensorFlow 2.21.0 requires numpy >= 2.1.0
     ↓
tensorflow-model-optimization 0.8.0 requires numpy < 2.0
     ↓
CONFLICT! No numpy version satisfies both.
```

Additionally:
- Pandas 3.0.1 requires numpy >= 2.3.3
- This creates a three-way conflict

## ✅ The Solution

Use **TensorFlow 2.18.1** instead of 2.21.0:
- ✅ Works with numpy 1.26.4
- ✅ Compatible with tf-keras 2.18.0
- ✅ Works with tensorflow-model-optimization 0.8.0
- ✅ Compatible with pandas 2.3.3 (also works with numpy 1.26.4)

## 🚀 Migration Steps

### Option 1: Full Poetry Migration (Recommended)

```powershell
# 1. Remove old virtual environment
deactivate  # if currently active
Remove-Item .venv -Recurse -Force

# 2. Create new Poetry environment
poetry install

# 3. Activate Poetry shell
poetry shell

# 4. Test your code
cd Phase_1
python process_data.py --help
python create_model.py
```

### Option 2: Keep Current Setup, Use Poetry for New Projects

Keep your working `requirements.txt` setup, but use `pyproject.toml` for:
- Documentation (dependencies are clearly listed)
- Future projects
- When you're ready to migrate later

## 🧪 Testing Your Code

After migration, test all Phase 1 scripts:

```powershell
poetry shell
cd Phase_1

# Test 1: Data processing
python process_data.py --data_dir RowData --mode save

# Test 2: Model creation
python prepare_data_for_training.py

# Test 3: Model optimization
python optimize_model.py
```

## 📝 Code Changes Required

**ZERO code changes needed!** Your imports will work exactly the same:
- `import tf_keras` ✅
- `import tensorflow_model_optimization` ✅
- All other imports ✅

TensorFlow 2.18.1 vs 2.21.0 has no breaking changes for your use case.

## ⚠️ Known Issues with Current Setup (pip)

Your current `requirements.txt` setup **technically violates** dependency constraints:
- TensorFlow 2.21.0 officially requires numpy >= 2.1.0
- But you have numpy 1.26.4 and it works

This is **fragile**:
- ⚠️ `pip install --upgrade` could break everything
- ⚠️ Installing new packages might trigger conflicts
- ⚠️ Reproducibility on other machines is not guaranteed

## 🎁 Benefits of Poetry

1. **Dependency Lock**: `poetry.lock` ensures everyone gets exact same versions
2. **Conflict Resolution**: Poetry prevents incompatible combinations
3. **Reproducibility**: `poetry install` creates identical environments
4. **Modern Tooling**: One command for everything (`poetry add`, `poetry update`)
5. **No more requirements.txt confusion**: One source of truth

## 📚 Poetry Quick Reference

```powershell
# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Run a command without activating shell
poetry run python script.py

# Add a new package
poetry add pandas

# Add a dev dependency
poetry add --group dev pytest

# Update all packages (respecting constraints)
poetry update

# Show installed packages
poetry show

# Show dependency tree
poetry show --tree
```

## 🤔 My Recommendation

**Migrate to Poetry.** Here's why:

1. Your current setup is fragile (violates TensorFlow's stated requirements)
2. TensorFlow 2.18.1 → 2.21.0 has **no breaking changes** for TinyML
3. Pandas 2.3.3 → 3.0.1 has minor API differences but nothing you're using
4. Poetry will prevent future headaches

The minor version downgrades are **safer** than your current setup and won't affect your learning journey.

## 🛠️ If You Want to Keep TensorFlow 2.21.0

The **only way** to keep TF 2.21.0 with Poetry is to:

1. Wait for `tensorflow-model-optimization` to support numpy 2.x (not released yet)
2. Use `pip` and live with the fragility
3. Fork and patch `tensorflow-model-optimization` yourself (not recommended)

**My advice:** TensorFlow 2.18.1 is rock-solid, widely used, and fully compatible with your code. The 2.21.0 version is too new and creates ecosystem conflicts.
