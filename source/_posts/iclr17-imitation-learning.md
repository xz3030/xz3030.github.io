---
title: >-
  【DRL论文阅读】Third-Person Imitation Learning 和前作Generative Adversarial Imitation
  Learning
tags: drl
categories: drl
date: 2017-04-14 00:10:56
---


# 前言

　　最近终于项目有一定进展，可以有时间看paper啦，撒花~  这个系列扫一下ICLR2017关于DRL的文章，希望能有些insights。

　　首先说个题外话，ICLR的审稿流程真是可怕，reviewer和author来回好几轮，工作量足够赶上一个journal，没有金刚钻是不敢揽这个瓷器活的。DRL的圈子被Deepmind和OpenAI把持，外人想进去很难，不过现阶段用DRL去做一些其他应用的道路还是很宽的，无论是写paper还是工业界都有着很大的价值，值得follow。

---

![](/images/imitation_learning_paper/iclr_imitation_learning_title.png)
> 解决的问题：第三人称视角模仿学习
使用方法：GAN、RL (TRPO)
亮点：GAN与RL结合、模仿学习应用场景扩展

<!--more-->

---

# 文章简介

　　第一篇文章讲一下 ["Third-Person Imitation Learning"](https://arxiv.org/pdf/1703.01703.pdf) [1]，出自OpenAI，挂着Pieter Abbeel和Ilya Sutskever两座大神的名字。正如题目所说，这篇文章讲的是以第三人称视角进行模仿学习(imitation learning)，即学习者通过观察老师的行为，进行模仿并最终实现任务，此过程中不需要学习者以第一人称的视角实际体验该任务，这个设定实际上很符合人类婴儿的学习过程。以后imitation learning可能就不需要agent亲自上场试验啦，想想看机器人仅靠眼睛看人类的行为即可做到模仿、学习，既感到fancy又觉得可怕。

![](/images/imitation_learning_paper/title.jpg "模仿学习")

　　推荐这篇文章的另外一个原因，其实是延续自其前作的一个很有意思的思想，即将Generative Adversarial Network (GAN)的思想引入Imitation Learning，把目前大火的 <font color="#FF0000">**GAN**</font> 和 <font color="#FF0000">**RL**</font> 有机的结合在一起。个人浅见，这是目前GAN在实际工业应用中最可能实现突破的一点。

　　理解这篇paper首先需要解释两个问题：1) 什么是imitation learning; 2) 理解其前作：["Generative Adversarial Imitation Learning"](http://papers.nips.cc/paper/6391-generative-adversarial-imitation-learning.pdf) (NIPS 16，作者是OpenAI的Jonathan Ho和Stefano Ermon) [2]。

---

# 模仿学习（Imitation Learning）

> 模仿学习，指学习者通过模仿专家的示范动作来完成任务的一种算法。

　　根据[2]的描述，其特点是：
* 学习者(learner)可以得到专家(expert)行为的轨迹或历史记录(trajactory)；
* 在学习过程中，learner无法从expert处追加查询更多的数据；
* Learner在学习过程中无法获得任何显式的奖励信号(reward signal)。

　　公式上，模仿学习一般提供专家的决策数据 {$\tau_E$}，每个决策包含着状态和动作序列 
$< s_1^i,a_1^i,s_2^i,a_2^2,...,s_n^i>$。将所有的状态-动作对抽离出来，以状态作为feature，动作作为label进行学习，并使得模型生成的状态-动作与输入轨迹尽可能相近。

　　实现imitation learning的方式主要有两种。最简单直接可以想到的，即用supervised learning的方法直接去拟合专家轨迹的状态-动作对，作为learner的策略(policy)。这种算法被称为 **Behavioral Cloning**。相关的supervised learning算法，可以想到的例子如RNN (LSTM)，Structural SVM, CRF等。由于传统的supervised learning不考虑agent和环境间的交互，Behavioral Cloning存在着序列行为中累积误差逐渐增大的问题。

![](/images/imitation_learning_paper/aggreg_error.png "误差累积问题")

　　另一种算法，即 ** IRL (Inverse Reinforcement Learning) **，根据专家的轨迹拟合出其做决策时的cost function，并根据这个cost function进行强化学习(Reinforcement Learning)以实现行为。该算法引入了agent和environment的交互，更适用于时间序列的学习问题。[2]中提到，由于IRL算法在内层循环中需要运行Reinforcement learning这一花费大量时间的过程，其scalability能力很有限，换句话说就是跑的太慢啦，不能解大型问题。

---

# GAIL (Generative Adversarial Imitation Learning)
　　这篇前作[2]解决的问题，即调过IRL，直接使用RL中的policy gradient算法学习专家的策略。

## IRL 目标函数解释
　　Inverse Reinforcement Learning，顾名思义，是RL的反过程，即通过行为体某策略的动作轨迹，反推行为体做决策时使用的奖励函数reward或代价函数cost。首先回顾一下符号定义：
- $\mathcal{S}$表示状态集，$\mathcal{A}$表示动作集；
- 状态转移$s\rightarrow s'$由环境中的转移概率$P(s'|s,a)$决定；
- 专家策略由$\pi_E$表示，学习者策略由$\pi$表示；
- 定义cost function $c(s,a)$，其目的是区分专家决策和学习者的决策，也就是说$c(s,a)$会倾向于给learner的策略比较高的cost，而给专家的策略低的cost；
- 定义 $ \mathbb{E}_{\pi}[c(s,a)] \triangleq \mathbb{E}[\sum_{t=0}^\infty \gamma^t c(s_t,a_t)] $，描述在策略$\pi$下的total discounted cost之和的期望，类似于RL里的value function.

　　在此定一下，IRL的目标函数可以定义为:

> $$ \max_{c\in \mathbb{C}} \big(\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_{\pi} [c(s,a)] \big) - \mathbb{E}_{\pi_E} [c(s,a)] $$

　　OK，下面我们来解释一下这个objective function。从外向内看，
- 首先，cost function $c$的作用是建立一种reward(cost)机制，使得其可以分别expert trajactories和learner trajactories。故公式的最外层是对c求max cost，即Inverse RL里由行为轨迹反求reward(cost)的过程；
- 向内一层，在第一个括号之内，是优化learner策略的过程，即求策略$\pi$，使其可最小化长期cost function；
- 最后，$H(\pi)$ 定义为 * $\gamma$-discounted casusal entropy*，$H(\pi)\triangleq \mathbb{E}\pi [-\log \pi(a|s)]$，其作用类似于supervised learning中s为feature，a为label的学习过程。

当代价函数$c$确定后，IRL在inner loop中就可以使用RL算法，利用学习到的cost function做为依据来得到最优策略：
$$ RL(c)=\arg\min_{\pi \in \Pi} -H(\pi)+\mathbb{E}_{\pi}[c(s,a)] $$

> 综上，IRL的求解过程为：计算learner的策略$\pi$使其尽量模仿expert的行为，并建立代价函数$c$使其尽量区分expert的真实轨迹和learner的模仿轨迹，重复迭代以上两步骤至收敛。

　　*等等，这是不是看起来很熟悉？很像GAN里面的印钞机和验钞机有没有？！* 没错！GAIL这篇文章的想法，就是将IRL改写成GAN的形式，使用neural network的强大特征表征能力来提升Imitation Learning的性能。当然，这篇文章的另一个贡献是将GAN以及 Abbeel and Ng 和 Syed之前的两篇学徒学习(Apprenticeship learning)的算法从数学上统一在了一个框架中，得到了很漂亮的公式表达。

## 公式推导与重要结论

介绍完IRL的主要思路后，文章里做了一系列的数学推导，得到了一个很general的公式来表征模仿学习的问题。我们这里就只提一些重要结论啦，感兴趣的读者可以啃啃原文。

- 由于专家轨迹的量有限，IRL学习cost function $c$时很容易overfit，于是没什么可说的，加个regularizer $\psi$吧！于是IRL目标函数变为了
$$ IRL_{\psi}(\pi_E) = \arg\max_{c\in\mathbb{R}^{\mathcal{S}\times \mathcal{A}}} -\psi(c) + \big(\min_{\pi \in \Pi} -H(\pi) + \mathbb{E}_{\pi} [c(s,a)] \big) - \mathbb{E}_{\pi_E}[c(s,a)]$$

- 定义occupancy measure $\rho_{\pi}: \mathcal{S}\times\mathcal{A}\rightarrow\mathbb{R}$ 为 $\rho_{\pi}(s,a)=\pi(a|s)\sum_{t=0}^{\infty}\gamma^t P(s_t=s|\pi)$，表示策略$\pi$下状态-动作对$< s,a>$出现的频率。则使用IRL进行Imitation learning的全过程可以推导为：
$$ RL\circ IRL_{\psi}(\pi_E) = \arg\min_{\pi \in \Pi} - H(\pi)+\psi^*(\rho_{\pi} - \rho_{\pi_E})$$

　　上式说明，加入了约束项的IRL目标函数，等价于找到一个策略$\pi$，使得其状态-动作对的出现频率(occupancy measure)尽量接近专家生成的状态-动作对，“接近”的评价标准是$\psi^*$，即约束函数的共轭函数(convex conjugate)。

- 以上面的发现为基础，作者证明了学徒学习(Apprenticeship learning)的目标函数，实际上对应着一种特殊的约束函数$\psi$，即indicator function $\psi=\delta_{\mathcal{C}}$，$\delta_{\mathcal{C}}(c)=0$ if $c\in\mathcal{C}$, and $\infty$ otherwise。注意学徒学习的目标函数是：

$$ \min_{\pi} \max_{c\in\mathcal{C}} \mathbb{E}_{\pi} [c(s,a)] - \mathbb{E}_{\pi_E} [c(s,a)]$$

　　而使用了$\delta_{\mathcal{C}}$的目标函数为：

$$ \min_{\pi} -H(\pi)+ \max_{c\in\mathcal{C}}\mathbb{E}_{\pi} [c(s,a)] - \mathbb{E}_{\pi_E} [c(s,a)]$$

- 作者的另一篇前作证明了用梯度下降的方式求解上式，可以用一个两步交替迭代的方法去解，而其中的一步正是对应于RL里的<font color="#FF0000">**policy gradient**</font>。简单的推导如下：

　　设学习者用参数$\theta$做价值函数模拟，故策略可写作$\pi_{\theta}$。对$\theta$求导：
$$\nabla_{\theta} \max_{c\in\mathcal{C}} \mathbb{E}_{\pi_{\theta}}[c(s,a)] - \mathbb{E}_{\pi_{E}}[c(s,a)]$$
　　设两个辅助函数：$\hat{c}=\arg\max_{c\in\mathcal{C}}\mathbb{E}_{\pi_{\theta}}[c(s,a)] - \mathbb{E}_{\pi_{E}}[c(s,a)]$, 和 $Q_{\hat{c}}(s,a)=\mathbb{E_{\theta}}[\hat{c}(s,a)|s_0=s,a_0=a]$
　　之前的梯度可以转化为：
$$ Left=\nabla_{\theta}\mathbb{E}_{\pi_{\theta}}[\hat{c}(s,a)] = \mathbb{E}_{\pi_{\theta}}[\nabla_\theta\log \pi_{\theta}(a|s) Q_{\hat{c}}(s,a)]$$
　　其中最后一个等号用到了经典的policy gradient的推导，不熟悉的可以参考这篇 [Karpathy的blog](http://karpathy.github.io/2016/05/31/rl/)。

- 上述的两步迭代求解方法即：
**1. 使用当前的策略$\pi_{\theta_i}$在环境中采样，并依此用上式拟合修正的代价函数 $\hat{c}$ ;**
**2. 使用拟合出的代价函数 $\hat{c}$，根据采样的历史记录，使用policy gradient优化当前策略得到新策略$\pi_{\theta_{i+1}}$ .**

> 以上两步中，步骤1对应着找到最能区分expert和learner的cost function的作用，步骤2对应根据cost function，优化learner的策略。

## 引入GAN

　　OK，经过以上一段复杂的数学推导，终于到了在目标函数中引入GAN的时候了。回顾一下，上文分析了每一种约束函数$\psi$，都可以对应一种新的学徒学习的算法。作者证明了，GAN对应的约束函数是：

![](/images/imitation_learning_paper/equation13.png)

　　可以证明，其目标函数有如下形式：
$$\psi^*_{GA}(\rho_\pi - \rho_{\pi_E})-\lambda H(\pi) = \sup_{D\in (0,1)^{\mathcal{S}\times\mathcal{A}}} \mathbb{E}_{\pi}[\log(D(s,a))] + \mathbb{E}_{\pi_E}[\log(1-D(s,a))]-\lambda H(\pi)$$

　　于是终于，熟悉的式子出现啦~ 由上式可知，GAN对应的cost function $c(s,a)=\log D(s,a)$，即验钞机的验钞本领。当learner越厉害（可以以假乱真）时，$D(s,a)\rightarrow 1$，对应$c(s,a)$ 最大，算法约需要*“用力”*地找到一个可以区分expert 和 learner的cost function。

## GAIL算法流程
作者将GAN+Imitation Learning(IL)命名为GAIL（名字不错）。最终的算法流程其实非常简单，见下图，paper的关键其实在于之前的数学推导。

![](/images/imitation_learning_paper/algorithm_gail.png "GAIL算法框架")

　　算法很清晰，第一步用GAN的目标函数拟合cost function，对应于GAN的$D_w(s,a)$ (line4)；第二步根据cost function使用TRPO(Trust Region Policy Optimization，一种policy gradient的改进，用于训练时不容易跑飞)更新策略$\pi_\theta$。重复迭代至收敛即可。


## 实验与代码实现
　　OpenAI kindly提供了GAIL的代码，传上了github：https://github.com/openai/imitation 。时间原因我还没有自己跑哈，测试之后再更新这段。
　　
---

# Third-Person Imitation Learning
　　理解了GAIL，我们的这篇正文"Third-Person Imitation Learning"就很简单了。个人浅见，这篇文章的主要贡献在于make it possible去做第三人称模仿学习这件事，而算法本身更多的是在现有算法上添砖加瓦，创新点并不是很多。

![](/images/imitation_learning_paper/third_title.png "Third-Person Imitation Learning 网络结构")

　　本文基本的想法，即在GAIL的基础上，引入第三人称和第一人称之间的domain difference，用经典的Deep Domain Adaptation的手段来求解GAIL。具体来说，作者将GAN的网络分成了两部分，前一半低层layers用于提取特征（设其为$D_F$），在两个domain共用；后一半的高层layers则分别对应于expert domain的$D_R$ 和 区分expert和learner的分类器$D_D$。作者用了一个BP过程中翻转梯度方向的trick $\mathcal{G}$，使得$D_R, D_D, D_F$在优化目标中都取$\min$。最终的算法流程图和GAIL相比变化不大，作者同样使用了TRPO作为policy gradient的解法。

![](/images/imitation_learning_paper/algorithm_third.png "Third-Person Imitation Learning 算法框架")


---

# 相关paper推荐

由于本文实质上只要是在讲GAIL，故这里只推荐一些后续cite GAIL的文章。

- 最值得一看的是Deepmind的一篇文章["Connecting Generative Adversarial Networks
and Actor-Critic Methods"](https://arxiv.org/pdf/1610.01945.pdf)，文章中分析了 GAN 和 RL中的Actor-Critic(AC) 算法在方法论上的相似性，并详细分析了两种算法在实际应用中的 stabilizing 方法，至少可以刚做一个实现GAN和AC算法时的trick使用手册。

- 另外有一篇很有意思的paper，关于[multi-agent imitation learning](https://arxiv.org/pdf/1703.03121)，内容大致是让一群机器人学习现实中球员比赛时的行为，以理解multi-agent间的协同合作关系。

---

# 结论
　　这篇blog借着"Third-Person Imitation Learning"这篇文章，回顾了Imitation Learning的问题设定，及前作将GAN和RL结合起来做imitation learning的算法GAIL，并详细解释了部分的理论推导。

从这篇文章我们可以得到的take-away是：
- Imitation Learning作为目前高速发展的一个领域，其应用场景在不断的扩展。几年之后机器人的模仿、学习能力就能上一个台阶就说不定。
- 用IRL的方法做Imitation Learning，其特例的IRL过程可以演化为GAN的形式，通过迭代的方式，优化GAN得到代价函数，再根据代价函数使用policy gradient优化策略，直至收敛。
- GAIL的想法可以看到一定的CV和NLP领域中的应用，如视频/序列的生成。


---

** 最近访问 ** 

<div class="ds-recent-visitors"
    data-num-items="36"
    data-avatar-size="42"
    id="ds-recent-visitors">
</div>