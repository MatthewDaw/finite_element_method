def optimize_model(self, actor_optimizer, critic_optimizer):
    """Optimize the model."""

    if len(self.reply_memory) < self.config.optimizer.batch_size:
        return

    transitions = self.reply_memory.sample(self.config.optimizer.batch_size)
    batched_transition = BatchedTransition(
        state=torch.stack([t.state for t in transitions]),
        action=torch.stack([t.state_choice_output for t in transitions]),
        next_state=torch.stack(
            [t.next_state if t.next_state is not None else torch.zeros_like(t.state) for t in transitions]),
        reward=torch.tensor([t.reward for t in transitions], device=self.config.device),
        done=torch.tensor([t.next_state is None for t in transitions], device=self.config.device, dtype=torch.float)
    )

    # Critic update
    with torch.no_grad():
        next_actions = self.parameter_server.policy_net_1(batched_transition.next_state)
        next_q_values = self.parameter_server.critic_net(batched_transition.next_state, next_actions)
        target_q_values = batched_transition.reward + (
                    1 - batched_transition.done) * self.config.epsilon.gamma * next_q_values

    current_q_values = self.parameter_server.critic_net(batched_transition.state, batched_transition.action)
    critic_loss = F.mse_loss(current_q_values, target_q_values)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Actor update
    actions = self.parameter_server.policy_net_1(batched_transition.state)
    actor_loss = -self.parameter_server.critic_net(batched_transition.state, actions).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    self.data_handler.losses.append(critic_loss.item())
    if ((len(self.reply_memory) % 25) == 0):
        np.save(f"{self.config.save_dir}/{self.config.agent_params.prefix}losses.npy",
                self.data_handler.losses)

    return critic_loss.item(), actor_loss.item()
