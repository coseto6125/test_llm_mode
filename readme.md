```mermaid
graph TD
    A[開始] --> B[初始化狀態]
    B --> C[啟動Agent 2]
    B --> D[啟動Agent 3]
    C --> E{asyncio.wait}
    D --> E
    E -->|FIRST_COMPLETED| F[Lambda轉出點]
    F --> G[返回主線程]
    
    F -.->|不計入執行時間| H[背景線程處理]
    H -.-> I[等待其餘任務]
    I -.-> J[聚合處理]
    J -.-> K[生成最終輸出]
    K -.-> L[數據持久化]
    
    style C fill:#f9d5e5,stroke:#333
    style D fill:#f9d5e5,stroke:#333
    style F fill:#ffcc00,stroke:#333,stroke-width:2px
    style G fill:#d5f9e5,stroke:#333
    style H fill:#e6e6e6,stroke:#999,stroke-dasharray: 5 5
    style I fill:#e6e6e6,stroke:#999,stroke-dasharray: 5 5
    style J fill:#e6e6e6,stroke:#999,stroke-dasharray: 5 5
    style K fill:#e6e6e6,stroke:#999,stroke-dasharray: 5 5
    style L fill:#e6e6e6,stroke:#999,stroke-dasharray: 5 5
    ```
