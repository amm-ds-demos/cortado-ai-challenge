import json
from typing import Dict, Any, List
from deepeval import assert_test
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import AnswerRelevancyMetric, GEval
from agent_backend import CortadoAgent
from config import get_config


class AgentEvaluator:
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the evaluator with configuration and agent backend.
        Args:
            config (Dict[str, Any]): The configuration dictionary.
        """
        self.config = config
        self.agent_backend = CortadoAgent()

    def create_geval_metric(self, metric_info: Dict[str, Any]) -> GEval:
        """
        Create a GEval metric based on the provided metric information.
        Args:
            metric_info (Dict[str, Any]): Information about the metric.
        Returns:
            GEval: The created GEval metric.
        """
        return GEval(
            model=self.config["evaluation"]["evaluation_model"],
            name=metric_info["name"],
            criteria=metric_info["criteria"],
            evaluation_params=[
                LLMTestCaseParams[param] for param in metric_info["evaluation_params"]
            ],
            threshold=self.config["evaluation"]["correctness_threshold"],
        )

    def evaluate_answers(
        self, questions: List[str], expected_answers: List[str], responses: List[str]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate answers using relevancy and custom GEval metrics.
        Args:
            questions (List[str]): List of questions.
            expected_answers (List[str]): List of expected answers.
            responses (List[str]): List of actual responses from the agent.
        Returns:
            List[Dict[str, Any]]: List of evaluations for each question.
        """
        evaluations = []
        for i, question in enumerate(questions):
            test_case = LLMTestCase(
                input=question,
                actual_output=responses[i],
                expected_output=expected_answers[i],
                context=[],
            )

            answer_relevancy_metric = AnswerRelevancyMetric(
                threshold=self.config["evaluation"]["relevancy_threshold"],
                model=self.config["evaluation"]["evaluation_model"],
            )
            custom_metrics = [
                self.create_geval_metric(metric)
                for metric in self.config["evaluation"]["geval_metrics"]
            ]

            assert_test(test_case, [answer_relevancy_metric] + custom_metrics)

            evaluation = {
                "question": question,
                "expected_answer": expected_answers[i],
                "response": responses[i],
                "metrics": {
                    "relevancy": answer_relevancy_metric.score,
                    **{metric.name: metric.score for metric in custom_metrics},
                },
            }
            evaluations.append(evaluation)

        return evaluations

    def serialize_tool_output(self, output: Any) -> Dict[str, Any]:
        """
        Serialize the tool output.
        Args:
            output (Any): The tool output to serialize.
        Returns:
            Dict[str, Any]: The serialized tool output.
        """
        return {
            "content": output.content,
            "tool_name": output.tool_name,
            "raw_input": output.raw_input,
            "raw_output": (
                output.raw_output
                if isinstance(output.raw_output, str)
                else (
                    output.raw_output.response
                    if hasattr(output.raw_output, "response")
                    else str(output.raw_output)
                )
            ),
            "is_error": output.is_error,
            "source_nodes": (
                [
                    {"id": node.node.id_, "text": node.node.text}
                    for node in output.raw_output.source_nodes
                ]
                if hasattr(output.raw_output, "source_nodes")
                else []
            ),
        }

    def serialize_agent_response(self, response: Any) -> Dict[str, Any]:
        """
        Serialize the agent response.
        Args:
            response (Any): The agent response to serialize.
        Returns:
            Dict[str, Any]: The serialized agent response.
        """
        return {
            "response": response.response,
            "sources": [
                self.serialize_tool_output(source) for source in response.sources
            ],
        }

    def process_agent_responses(self, responses: List[Any]) -> Dict[str, Any]:
        """
        Process agent responses for serialization.
        Args:
            responses (List[Any]): List of agent responses.
        Returns:
            Dict[str, Any]: Processed agent responses.
        """
        data = [self.serialize_agent_response(response) for response in responses]
        return {"agent_responses": data}

    def save_to_json(self, data: Dict[str, Any], filename: str):
        """
        Save data to a JSON file.
        Args:
            data (Dict[str, Any]): The data to save.
            filename (str): The filename to save the data to.
        """
        with open(filename, "w") as f:
            json.dump(data, f, indent=4)

    def run_evaluation(self):
        """
        Run the evaluation process.
        """
        # Load questions and expected answers
        with open(self.config["evaluation"]["questions_file"], "r") as f:
            questions_data = json.load(f)

        questions = questions_data["questions"]
        expected_answers = questions_data["expected_answers"]

        # Process questions and evaluate answers
        responses = []
        for q in questions:
            try:
                response = self.agent_backend.process_question(q)
                responses.append(response)
            except Exception as e:
                responses.append(f"Error processing question '{q}': {str(e)}")

        print(responses)
        evaluations = self.evaluate_answers(
            questions,
            expected_answers,
            [i.response for i in responses if hasattr(i, "response")],
        )
        processed_responses = self.process_agent_responses(responses)
        self.save_to_json(
            processed_responses,
            self.config["evaluation"]["processed_agent_responses_file"],
        )

        with open(self.config["evaluation"]["evaluations_output_file"], "w") as f:
            json.dump(evaluations, f, indent=4)


if __name__ == "__main__":
    config = get_config()
    evaluator = AgentEvaluator(config)
    evaluator.run_evaluation()
