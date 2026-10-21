"""Source/config regression checks; no model imports, images or weights are read."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]


def module(path):
    return ast.parse((ROOT / path).read_text())


def definition(path, name):
    return next(n for n in module(path).body if getattr(n, "name", None) == name)


class DiffusionReleaseTests(unittest.TestCase):
    def test_public_configs_match_trainer_signature_and_stage_chain(self):
        trainer = definition(
            "src/diffusion/flow_matching/training_rapd.py", "RaPDTrainer"
        )
        init = next(n for n in trainer.body if getattr(n, "name", None) == "__init__")
        accepted = {a.arg for a in init.args.args} - {"self"}
        expected = [
            ("rapd_pretrain_256", 300000, 192, 2e-4, 16.0),
            ("rapd_pretrain_512", 400000, 64, 1e-4, 32.0),
            ("rapd_sft_512", 460000, 64, 5e-5, 32.0),
        ]
        previous = None
        for name, steps, batch, lr, shift in expected:
            cfg = yaml.safe_load((ROOT / f"configs_t2i/{name}.yaml").read_text())
            model = cfg["model"]
            args = model["diffusion_trainer"]["init_args"]
            self.assertFalse(set(args) - accepted)
            self.assertEqual(args["feat_loss_weight"], 0.5)
            self.assertEqual(args["timeshift"], shift)
            self.assertEqual(float(model["optimizer"]["init_args"]["lr"]), lr)
            self.assertEqual(cfg["trainer"]["max_steps"], steps)
            self.assertEqual(cfg["data"]["train_batch_size"], batch)
            self.assertEqual(model["ae"]["init_args"]["renderer_block_type"], "window_attention")
            if previous:
                self.assertTrue(cfg["finetune"])
                self.assertEqual(
                    cfg["ckpt_path"],
                    f"./rapd_work_dir/exp_{previous[0]}/epoch=0-step={previous[1]}.ckpt",
                )
            previous = name, steps
        mixture = cfg["data"]["train_dataset"]["init_args"]["long_caption"]
        self.assertTrue(mixture["enabled"])
        self.assertEqual(mixture["ratio"], 0.2)

    def test_legacy_plain_checkpoint_routes_to_mainline_modules(self):
        # Execute only load_ckpt with fake modules. There is no torch/model import.
        checkpoint = {
            "state_dict": {
                "denoiser.weight": "denoiser",
                "ema_denoiser.weight": "ema",
                "diffusion_trainer.proj.0.0.weight": "projection",
                "diffusion_trainer.renderer.old": "unused renderer",
                "diffusion_trainer.lpips_loss_fn.old": "unused perceptual model",
                "diffusion_trainer.z_mean": "unused stats",
            },
            "optimizer": {"param_groups": ["denoiser", "projection"]},
            "global_step": 300000,
        }

        class FakeModule:
            def state_dict(self, destination, prefix):
                pass

            def load_state_dict(self, state, strict):
                self.loaded = state
                self.strict = strict

        ns = {"torch": SimpleNamespace(load=lambda *a, **kw: checkpoint)}
        node = definition("main.py", "load_ckpt")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "load_ckpt", "exec"), ns)
        denoiser, ema, trainer = FakeModule(), FakeModule(), FakeModule()
        result = ns["load_ckpt"]("unused", denoiser, ema, trainer, "cpu")
        self.assertIs(result, checkpoint)
        self.assertEqual(denoiser.loaded, {"weight": "denoiser"})
        self.assertEqual(ema.loaded, {"weight": "ema"})
        self.assertEqual(trainer.loaded["proj.0.0.weight"], "projection")
        self.assertFalse(trainer.strict)
        # Extra legacy keys are passed to non-strict loading, not re-instantiated.
        cls = definition(
            "src/diffusion/flow_matching/training_rapd.py", "RaPDTrainer"
        )
        attrs = {
            n.attr for n in ast.walk(cls)
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
            and n.value.id == "self"
        }
        self.assertFalse(attrs & {"renderer", "lpips_loss_fn", "z_mean", "z_std"})

    def test_inference_builds_only_selected_network(self):
        import copy

        class FakeNetwork:
            def to(self, device):
                return self

        built = []

        def instantiate(*args):
            net = FakeNetwork()
            built.append(net)
            return net

        ns = {"instantiate_from_config": instantiate, "copy": copy}
        node = definition("main.py", "build_denoisers")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "build", "exec"), ns)
        build = ns["build_denoisers"]
        original, ema = build({}, "cpu", False)
        self.assertIsNone(original)
        self.assertIs(ema, built[-1])
        original, ema = build({}, "cpu", False, True)
        self.assertIs(original, built[-1])
        self.assertIsNone(ema)
        original, ema = build({}, "cpu", True)
        self.assertIs(original, built[-1])
        self.assertIsNot(original, ema)
        self.assertEqual(len(built), 3)

    def test_inference_checkpoint_routes_only_selected_weights(self):
        checkpoint = {"state_dict": {
            "denoiser.weight": "original",
            "ema_denoiser.weight": "ema",
            "diffusion_trainer.proj.0.0.weight": "training-only",
        }}

        class FakeNetwork:
            def state_dict(self, destination, prefix):
                pass

            def load_state_dict(self, state, strict):
                self.loaded = state

        ns = {"torch": SimpleNamespace(load=lambda *a, **kw: checkpoint)}
        node = definition("main.py", "load_ckpt")
        exec(compile(ast.Module(body=[node], type_ignores=[]), "load", "exec"), ns)
        for use_original in (False, True):
            net = FakeNetwork()
            ns["load_ckpt"]("unused", net if use_original else None,
                            None if use_original else net, None, "cpu")
            self.assertEqual(net.loaded, {"weight": "original" if use_original else "ema"})

    def test_inference_config_has_no_training_dependencies(self):
        cfg = yaml.safe_load((ROOT / "configs_t2i/inference.yaml").read_text())
        self.assertNotIn("diffusion_trainer", cfg["model"])
        self.assertNotIn("ema_tracker", cfg["model"])
        self.assertEqual(cfg["model"]["decode_overscan_pad"], 8)
        main = module("main.py")
        # Teacher and trainer imports must stay inside the training-only branch.
        imports = [n for n in main.body if isinstance(n, ast.ImportFrom)]
        self.assertFalse(any(n.module in {
            "src.models.encoder", "src.diffusion.flow_matching.training_rapd"
        } for n in imports))
        entry = definition("main.py", "main")
        branches = [n for n in entry.body if isinstance(n, ast.If)
                    and isinstance(n.test, ast.Name) and n.test.id == "is_training"]
        self.assertTrue(any(any(isinstance(n, ast.ImportFrom)
                               and n.module == "src.models.encoder"
                               for n in ast.walk(branch)) for branch in branches))

    def test_sampler_call_contract(self):
        ns = {}
        for name in ("sampler_forward", "sampler_forward_with_trajectories"):
            node = definition("main.py", name)
            exec(compile(ast.Module(body=[node], type_ignores=[]), name, "exec"), ns)
        calls = []

        def sampler(*args, **kwargs):
            calls.append((args, kwargs))
            return "result"

        inputs = ("net", "noise", "condition", "uncondition")
        self.assertEqual(ns["sampler_forward"](sampler, *inputs), "result")
        self.assertEqual(calls[-1], (inputs, {}))
        ns["sampler_forward_with_trajectories"](sampler, *inputs)
        self.assertEqual(calls[-1], (inputs, {"return_x_trajs": True}))


if __name__ == "__main__":
    unittest.main()
