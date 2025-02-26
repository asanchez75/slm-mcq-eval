import concurrent.futures

def generate_question_parallel(item, chain):
    try:
        generated_question = chain.invoke({"query": item['content']})
        return {
            "folder": item['folder'],
            "content": item['content'],
            "question": generated_question
        }
    except Exception as e:
        print(f"Error occurred for item in folder {item['folder']}: {e}")
        return None

def run_in_parallel(dataset, chain, max_workers=10):
    questions = [None] * len(dataset)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(generate_question_parallel, item, chain): index
            for index, item in enumerate(dataset)
        }
        processed_count = 0
        for future in concurrent.futures.as_completed(future_to_index):
            index = future_to_index[future]
            result = future.result()
            if result is not None:
                questions[index] = result
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"{processed_count} samples processed...")
    return [q for q in questions if q is not None]