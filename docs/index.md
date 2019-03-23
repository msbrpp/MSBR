## Abstract

This paper presents a semi-supervised learning framework to train a keypoint pose detector using multiview image streams given the limited number of labeled data (typically <4%). We leverage the complementary relationship between multiview geometry and visual tracking to provide three types of supervisionary signals for the unlabeled data: (1) pose detection in one view can be used to supervise that of the other view as they must satisfy the epipolar constraint; (2) pose detection must be temporally coherent in accordance with its optical flow; (3) the occluded keypoint from one view must be consistently invisible from the near views. We formulate the theory of multiview supervision by registration and design a new end-to-end neural network that integrates these supervisionary signals in a differentiable fashion to incorporate the large unlabeled data in pose detector training. The key innovation of the network is the ability to reason about the visibility/occlusion, which is indicative of the degenerate case of detection and tracking. Our resulting pose detector shows considerable outperformance comparing the state-of-the-art pose detectors in terms of accuracy (keypoint detection) and precision (3D reconstruction). We validate our approach with challenging realworld data including the pose detection of non-human species such as monkeys and dogs.

Preprint: [arxiv](https://arxiv.org/abs/1811.11251)

## Framework of MSBR
<img src="imgs/framework.png" width="640" height="400" alt="hi" class="inline"/>
<img src="imgs/mouse_epi.gif" width="300" height="300" alt="hi" class="inline"/>

## Evaluation 
<video src="vids/msbr.mp4" width="640" height="400" controls preload></video>

### Non-human (Customed keypoints)

### Human joints






Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).


### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
