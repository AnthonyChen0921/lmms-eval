import datasets
from typing import Callable, Iterable, Optional
from abc import ABC, abstractmethod


class ContextObject(ABC):
    @abstractmethod
    def get_text(self):
        raise NotImplementedError

    def __str__(self):
        return self.get_text()


class QAPairs(ContextObject):
    def __init__(self, question: str, answer: Optional[str] = None, delimiter="\n", role_question: str = "USER: ", role_answer: str = "ASSISTANT: "):
        self.question = question
        self.answer = answer
        self.delimiter = delimiter
        self.role_question = role_question
        self.role_answer = role_answer

    def get_text(self):
        if self.answer is None:
            return self.role_question + self.question + self.delimiter
        else:
            return self.role_question + self.question + self.delimiter + self.role_answer + self.answer


class LazyLoadedImages(ContextObject):
    def __init__(self, data_frame, index, doc_to_visual: Callable, image_tokens="<image>"):
        self.data_frame: datasets.Dataset = data_frame
        self.index = index
        self.image_lens = None
        self.images = None
        self.doc_to_visual = doc_to_visual
        self.image_tokens = image_tokens

    def get_images(self, lazy_save=False):
        if self.images is not None:
            return self.images
        images = self.doc_to_visual(self.data_frame[self.index])
        self.image_lens = len(images)
        if lazy_save:
            self.images = images
        return images

    def get_num_images(self, lazy_save=False):
        if self.image_lens is None:
            images = self.get_images(self.doc_to_visual)
            if lazy_save:
                self.images = images
            self.image_lens = len(images)
        return self.image_lens

    def clear(self, clear_all=False):
        self.images = None
        if clear_all:
            self.image_lens = None

    def get_text(self, lazy: bool = True):
        if lazy:
            return self.image_tokens
        else:
            return " ".join([self.image_tokens] * self.get_num_images())


class Context(object):
    def __init__(self, task, few_shot_delimiter: str = "\n\n", target_delimiter: str = "\n", description=None):
        self.task = task
        self.config = task._config

        self.doc_to_visual = self.task.doc_to_visual
        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.target_delimiter = target_delimiter
        self.few_shot_delimiter = few_shot_delimiter

        self.contexts = []

        if description:
            self.add_description(description)

    def add_description(self, description):
        self.contexts = [description] + self.contexts

    def get_question(self, doc):
        text = self.doc_to_text(doc)
        return text if (self.doc_to_choice is None or isinstance(text, str)) else self.doc_to_choice(doc)[text]

    def get_target(self, doc):
        return (
            str(self.doc_to_target(doc)[0])
            if type(self.doc_to_target(doc)) is list
            else self.doc_to_target(doc) if (self.config.doc_to_choice is None or type(self.doc_to_target(doc)) is str) else str(self.doc_to_choice(doc)[self.doc_to_target(doc)])
        )

    def add_in_context_example(self, doc, data_frame=None, index=None):
        question = self.get_question(doc)
        if data_frame and index:
            visual = LazyLoadedImages(data_frame, index, self.doc_to_visual)
        else:
            visual = None
        target = self.doc_to_target(doc)
        if visual:
            self.contexts.append(visual)
        self.contexts.append(QAPairs(question, target, self.target_delimiter))
        self.contexts.append(self.few_shot_delimiter)

    def add_question(self, doc, data_frame=None, index=None):
        question = self.get_question(doc)
        if data_frame and index:
            visual = LazyLoadedImages(data_frame, index, self.doc_to_visual)
        else:
            visual = None
        if visual:
            self.contexts.append(visual)
        self.contexts.append(QAPairs(question))
        # self.contexts.append(self.target_delimiter)

    def get_text(self, *, image_tokens="<image>", lazy=True):
        texts = []
        vision = []
        already_have_images = False
        for context in self.contexts:
            if isinstance(context, str) and image_tokens in context:
                already_have_images = True
                break
        if already_have_images:
            image_tokens = ""
        for context in self.contexts:
            if isinstance(context, LazyLoadedImages):
                if isinstance(image_tokens, str):
                    if lazy:
                        texts.append(image_tokens)
                    else:
                        now_vision = context.get_images(self.doc_to_visual)
                        vision.extend(now_vision)
                        texts.append(image_tokens * len(now_vision))
                else:
                    texts.append(image_tokens(context))
            else:
                texts.append(str(context))
        if lazy:
            return "".join(texts)
        else:
            return "".join(texts), vision

    def get_visions(self):
        return sum([context.get_images(self.doc_to_visual) for context in self.contexts if isinstance(context, LazyLoadedImages)], start=[])

    def extend(self, context):
        if isinstance(context, list):
            self.contexts.extend(context)
        elif isinstance(context, Context):
            self.contexts.extend(context.contexts)
        else:
            raise ValueError(f"Cannot extend context with object of type {type(context)}")

    def append(self, context):
        self.contexts.append(context)

    def __str__(self):
        return self.get_text()

    def __lt__(self, other):
        if not isinstance(other, Context):
            return NotImplemented
        return self.get_text() < other.get_text()


class FewShotDataset(object):
    def __init__(self, dataset=None, *, dataset_path: str = None, dataset_name: str = None, split: str = None, dataset_kwargs: dict = None, same_as_eval: bool = False):
        if dataset is not None and (dataset_path is not None or dataset_name is not None or split is not None or dataset_kwargs is not None):
            raise ValueError("Cannot provide both `dataset` and other dataset arguments!")
        self.dataset_path = dataset_path
        self.dataset_name = dataset_name
        self.split = split
        self.dataset = dataset
        self.dataset_kwargs = dataset_kwargs if dataset_kwargs is not None else {}
        self.same_as_eval = same_as_eval
        self.fewshot_indices = None

    def get_dataset(self) -> datasets.Dataset:
        if self.dataset is None:
            self.dataset = datasets.load_dataset(path=self.dataset_path, name=self.dataset_name, split=self.split, download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS, **self.dataset_kwargs)
            if self.fewshot_indices:
                self.dataset = self.dataset.select(self.fewshot_indices)
        return self.dataset

    def sample(self, n, rnd):
        indices = rnd.sample(range(len(self.get_dataset())), n)
        return indices, self.get_dataset().select(indices)

    def __getitem__(self, item):
        return self.get_dataset()[item]


class ContextSampler:
    def __init__(self, docs: FewShotDataset, task, fewshot_indices=None, rnd=None) -> None:
        self.rnd = rnd
        assert self.rnd, "must pass rnd to FewShotSampler!"

        self.task = task
        self.config = task._config

        self.target_delimiter = self.config.target_delimiter
        self.fewshot_delimiter = self.config.fewshot_delimiter

        self.doc_to_text = self.task.doc_to_text
        self.doc_to_target = self.task.doc_to_target
        self.doc_to_choice = self.task.doc_to_choice

        self.docs: FewShotDataset = docs  # HF dataset split, provided by task._fewshot_docs()
        if fewshot_indices:  # subset few-shot docs from
            self.docs.fewshot_indices = fewshot_indices

    def get_context(self, doc, num_fewshot) -> Context:
        # draw an extra fewshot sample if using same split as evaluating on
        n_samples = num_fewshot + 1 if self.docs.same_as_eval else num_fewshot

        # draw `n_samples` docs from fewshot_docs
        indices, fewshotex = self.sample(n_samples)

        # get rid of the doc that's the one we're evaluating, if it's in the fewshot
        # TODO: should we just stop people from using fewshot from same split as evaluating?
        selected_docs = [(idx, x) for idx, x in zip(indices, fewshotex) if x != doc][:num_fewshot]

        labeled_examples = Context(self.task, self.fewshot_delimiter, self.target_delimiter)

        for idx, doc in selected_docs:
            labeled_examples.add_in_context_example(doc, self.docs, idx)

        return labeled_examples

    def sample(self, n):
        """
        Draw `n` samples from our fewshot docs. This method should be overridden by subclasses.
        """

        return self.docs.sample(n, self.rnd)


class FirstNSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        Draw the first `n` samples in order from the specified split.
        Used for tasks with "canonical" ordered fewshot examples, such as MMLU and CMMLU.
        """
        assert n <= len(self.docs), f"Error: number of fewshot samples requested exceeds the {len(self.docs)} that are available."
        return self.docs[:n]


class BalancedSampler(ContextSampler):
    def sample(self, n) -> None:
        """
        TODO: this should return approximately class-balanced samples from our fewshot examples.
        TODO: what order should they be in? maybe random?
        """

        pass


class ManualSampler(ContextSampler):
    def sample(self, n) -> None:
        """ """
        pass


SAMPLER_REGISTRY = {
    "default": ContextSampler,
    "first_n": FirstNSampler,
}


def get_sampler(name):
    try:
        return SAMPLER_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Attempted to use contextsampler '{name}', but no sampling strategy for this name found! Supported model names: {', '.join(SAMPLER_REGISTRY.keys())}")
