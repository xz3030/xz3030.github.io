---
title: >-
  【DRL论文阅读】Multi-Agent Cooperation and the Emergence of (Natural) Language
  Learning
tags: drl
categories: drl
date: 2017-04-15 21:57:39
---


![](/images/multi_agent_conversation/title.png)

> 解决的问题：两个机器人之间玩“你画我猜”
使用方法：RL (Reinforce), t-SNE可视化分析
亮点：很有意思的应用，让两个agent玩游戏，观察agents间的交互方式

<!--more-->

---

# 文章简介

　　这两天尝试啃了几篇非常理论的Paper之后，发现还是找一些更简单、应用更有意思的paper看起来比较轻松~ 于是今天来聊一下一篇很有意思的论文：["Multi-Agent Cooperation and the Emergence of (Natural) Language"](https://arxiv.org/pdf/1612.07182.pdf)，来自Deepmind和Facebook AI Research的合作。

　　这篇文章大致做了这么一件事情：让两个agent分别扮演 *描述者* 和 *猜题者* 两个角色共同玩一个游戏。描述者看到两张不同concept的图片（分别是目标图片和干扰图片），并得知哪张图是目标图片。游戏的目标是让描述者提供一个词（词库由两个agent提前商量好，类似于暗号），使得猜题者可以通过这个词成功的分别两张图片，找到目标图片。设定有点像“你画我猜”或者是“看动作猜词语”这种游戏。

　　之前我们在研究中更多地都是focus在supervised learning上，而却忽视了通往通用AI之路的一个重要问题：AI和他人协同、交互的能力。这篇文章用很简单的方法，测试了AI间协作交互的能力，并试图分析这种交互，乃至干预AI间交互的形式变成人类可理解的语言，非常有意思。

> AI能否学到如何与同伴通信并互相理解？

---

# 游戏流程与算法

　　这篇文章的想法非常简单，就是一个常用的Reinforce搞定。于是我们重点来看一下作者是如何构造的游戏使得AI间可以协同合作。文章里使用了一种叫做 *referential games* 的设定：

- 首先建立一个图片集$i_1,...,i_N$，它们分别来自于463个基础概念（concept，如猫、苹果、汽车等）、20个大类，每个concept从ImageNet库里拉了100张图片（*论万能的ImageNet...*）。每一次游戏开始时，系统随机的选择两个concept，并分别选择一张图片 $(i_L,i_R)$，把其中的一个作为目标 $t\in $\{$ L,R $\}，另一个作为伪装。

- 游戏中共有两个角色：*描述者* 和 *猜题者* 。他们都能看到两张输入图片，但是只有描述者知道哪个是目标图片，即收到了输入 $\theta_S(i_L,i_R,t)$。

- 描述者和猜题者共同维护一个暗号集 $V$（size为$K$）。描述者选择 $V$ 中的一个暗号 $s$ 传给猜题者。描述者的策略记为：$s(\theta_S(i_L,i_R,t)) \in V$。

- 猜题者并不清楚哪张图是目标图片，于是他根据描述者传来的信息和两张图片本身来猜测目标图片。猜题者的策略记为：$r\big(i_L,i_R,s(\theta_S(i_L,i_R,t))\big) \in $\{$ L,R $\} 。

- 如果猜题者猜对了目标图片，即$r\big(i_L,i_R,s(\theta_S(i_L,i_R,t))\big) = t$，描述者和猜题者同时加一分（胜利）；否则都不加分（失败）。

　　作者建立了两种简单的图像embedding方式，分别是（如下图所示）
1) 直接将 VGG的softmax层输出（或fc层输出）embedding。称为 *agnostic* 方法。
2) 将输出再过一层2x1的卷积层降维，把目标图和伪装图的输出混合起来，最后再embedding。称为 *informed* 方法。

![](/images/multi_agent_conversation/architecture.png)

　　建立了policy和reward，玩游戏的问题自然地被转化成了RL过程，作者使用了Reinforce（最简单的policy gradient方法之一）算法来求解。结果很愉快，所有的agent都在10k轮之内成功收敛了，且答对率接近100%！其中，*informed* 方法相比于 *agnostic* 方法由于使用了更多的信息，可以收敛更快并更准。

　　当然了，这个游戏本身其实挺简单的（concept之间差的挺大的，imagenet的vgg特征也足够强），所以performance倒并不是最重要的，更有意义的是去分析 agent之间到底是如何(chuan)交(an)互(hao)的，以为这种交互方式是否能被我们人类所理解？

> 试想如果有一天 AI 之间发明了一种特殊的暗号，人类完全破译不能。。。 orz 

---

# 了解AI间的交互方式

　　为了潜入AI的脑内，分析其与同伴打招呼的方式，作者设计了几种trick来证明或可视化他们的假设。下面我们简单看看作者们的发现：

> AI 有丰富的描述语言，并能抓住重点。

　　作者的第一个发现，就是 *informed sender* 使用了较多种的词语作为暗号来通信，且这些暗号中至少有很大一部分代表着不同的意义。同时，这些暗号的意义与人类定义的20种大类有着一定的相关性（最高的一种算法的聚类purity有46%）。
　　另一方面，*agnostic sender* 弱弱地只使用过两个词作为暗号。作者肉眼看的结果是这个暗号类似于 “能喘气的 vs. 不动换的” (living-vs-non-living)，也确实符合imagenet“动物园”的特性。

> 可以通过“去常识化”使得 AI 的词语与人类的认知进一步相似

　　用博弈论的调调来说，常识指所有人都知道的，以及所有人都知道所有人都知道的。。。把输入中的常识分量去掉，可以使得agent学会更本质的东西。作者的实现方法很tricky，给描述者和猜题者看了两张很相近但确不完全相同的图片（比如目标concept是狗，两个agent一个看到的是吉娃娃，另一个则看到的是波士顿小狗）。结果是agent的暗号与人类定义的类别更相近了。这点其实我没太理解。。。

![](/images/multi_agent_conversation/tsne.png "去常识化之前（左）和之后（右）的AI暗号聚类示意图，同样颜色对应着人类标识的相同大类")

> 可以通过加入监督学习的方式使得 AI 用人类的词语通信

　　最后，当然我们可以用监督学习的方式，显式地让 AI 学会人类的语言来通信。作者在最后一个实验中，迭代地进行增强学习来优化agent和监督学习来让agent学会人类语言。评价时，作者雇了一群真正的人来当猜题者，并由AI生成的词语（监督学习中对应着imagenet的label）作为提示。结果是68%的人能猜到正确的图片（心疼这些人几秒钟。。。），似乎比预想的要低一些，但至少能大幅超过random guess。

---

# 结论

　　总体来说，这篇文章提出了一个很有意思的应用，依此研究了AI间交互的可能性与其形式。理论算法和实验的结果其实都并不是很强，相信还有很大的提升空间。

---

** 最近访问 ** 

<div class="ds-recent-visitors"
    data-num-items="36"
    data-avatar-size="42"
    id="ds-recent-visitors">
</div>