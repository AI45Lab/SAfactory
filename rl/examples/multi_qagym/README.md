# Multi-QAGym RL Example

This example runs one shared QAGym rollout with two trainable policies:

- `attacker_policy`: generates adversarial prompts.
- `defender_policy`: answers the attacker prompt safely.

The env owns the turn order:

```text
attacker -> defender -> judge reward
```

The rollout is shared, but each policy is trained by its own Slime trainer.

## Run

Start the buffer server yourself:

```bash
./run_buffer_server.sh
```

Then, for a 4-GPU machine, use the launcher for the two trainers:

```bash
bash ./run_train_4gpu.sh
```

It starts shared Ray, the defender trainer, and then the attacker trainer that
owns the shared rollout. It expects the buffer server to already be running.

To run each trainer manually, start the non-owner defender trainer first so its
policy endpoint is available:

```bash
./run_defender_generator.sh
```

Start the attacker trainer as rollout owner:

```bash
./run_attacker_generator.sh
```

The attacker trainer starts the shared rollout. Both trainers read from the same
buffer server, filtered by `policy_id`.

## Notes

- `env.sh` is a forced config file. Edit it directly if ports, policy IDs, or GPU layout need to change.
- The default env config is `env/multi_qagym/multi_qagym_env.yaml`.
- The judge endpoint in the env config is a placeholder and should be set to a real judge model endpoint before running.
