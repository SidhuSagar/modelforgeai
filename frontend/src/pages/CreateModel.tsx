// src/pages/CreateModel.tsx
import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Progress } from "@/components/ui/progress";
import { useToast } from "@/hooks/use-toast";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import {
  FileText,
  MessageSquare,
  BookOpen,
  ArrowRight,
  ArrowLeft,
  Sparkles,
  Settings,
  Brain,
  Upload,
  Database,
  CheckCircle,
  Download,
  Rocket,
  Check,
  LucideIcon,
  Network,
  Layers,
  Zap,
} from "lucide-react";

// Types
interface Step {
  id: number;
  name: string;
  description: string;
}

interface ModelConfig {
  task?: string;
  modelType?: string;
  dataset?: File | string;
  preprocessing?: {
    textColumn?: string;
    labelColumn?: string;
    trainSplit?: number;
  };
  training?: {
    modelName?: string;
    budget?: number;
    async?: boolean;
  };
  modelInfo?: any;
}

interface TaskCardProps {
  icon: LucideIcon;
  title: string;
  description: string;
  selected?: boolean;
  onClick?: () => void;
}

interface StepIndicatorProps {
  steps: Step[];
  currentStep: number;
}

interface TaskSelectionProps {
  config: ModelConfig;
  updateConfig: (updates: Partial<ModelConfig>) => void;
  onNext: () => void;
}

interface ModelTypeSelectionProps {
  config: ModelConfig;
  updateConfig: (updates: Partial<ModelConfig>) => void;
  onNext: () => void;
  onBack: () => void;
}

interface DatasetSelectionProps {
  config: ModelConfig;
  updateConfig: (updates: Partial<ModelConfig>) => void;
  onNext: () => void;
  onBack: () => void;
}

interface PreprocessingConfigProps {
  config: ModelConfig;
  updateConfig: (updates: Partial<ModelConfig>) => void;
  onNext: () => void;
  onBack: () => void;
}

interface TrainingConfigProps {
  config: ModelConfig;
  updateConfig: (updates: Partial<ModelConfig>) => void;
  onNext: () => void;
  onBack: () => void;
}

interface ModelTrainingProps {
  config: ModelConfig;
  onNext: () => void;
  onBack: () => void;
  updateConfig: (updates: Partial<ModelConfig>) => void;
}

interface ModelTestingProps {
  config: ModelConfig;
  onBack: () => void;
}

// Step Indicator Component
const StepIndicator = ({ steps, currentStep }: StepIndicatorProps) => {
  return (
    <nav aria-label="Progress">
      <ol className="flex items-center justify-between">
        {steps.map((step, stepIdx) => (
          <li
            key={step.name}
            className={cn(
              "relative",
              stepIdx !== steps.length - 1 ? "pr-8 sm:pr-20 flex-1" : ""
            )}
          >
            {step.id < currentStep ? (
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-primary" />
                </div>
                <div className="relative flex h-10 w-10 items-center justify-center rounded-full bg-primary shadow-[var(--glow-primary)]">
                  <Check className="h-5 w-5 text-primary-foreground" />
                </div>
              </>
            ) : step.id === currentStep ? (
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-border" />
                </div>
                <div className="relative flex h-10 w-10 items-center justify-center rounded-full border-2 border-primary bg-background shadow-[var(--glow-primary)]">
                  <span className="text-primary font-semibold">{step.id}</span>
                </div>
              </>
            ) : (
              <>
                <div className="absolute inset-0 flex items-center" aria-hidden="true">
                  <div className="h-0.5 w-full bg-border" />
                </div>
                <div className="relative flex h-10 w-10 items-center justify-center rounded-full border-2 border-border bg-card">
                  <span className="text-muted-foreground">{step.id}</span>
                </div>
              </>
            )}
            <span className="absolute -bottom-6 left-1/2 -translate-x-1/2 text-xs font-medium whitespace-nowrap text-muted-foreground">
              {step.name}
            </span>
          </li>
        ))}
      </ol>
    </nav>
  );
};

// Task Card Component
const TaskCard = ({
  icon: Icon,
  title,
  description,
  selected,
  onClick,
}: TaskCardProps) => {
  return (
    <Card
      className={cn(
        "p-6 cursor-pointer transition-all duration-300 hover:scale-105 hover:shadow-[var(--glow-primary)] border-2",
        selected
          ? "border-primary bg-gradient-to-br from-card to-secondary shadow-[var(--glow-primary)]"
          : "border-border bg-card hover:border-primary/50"
      )}
      onClick={onClick}
    >
      <div className="flex flex-col items-center text-center space-y-4">
        <div
          className={cn(
            "p-4 rounded-full transition-colors",
            selected
              ? "bg-primary text-primary-foreground"
              : "bg-secondary text-foreground"
          )}
        >
          <Icon className="h-8 w-8" />
        </div>
        <div>
          <h3 className="text-xl font-semibold mb-2">{title}</h3>
          <p className="text-sm text-muted-foreground">{description}</p>
        </div>
      </div>
    </Card>
  );
};

// Task Selection Step
const TaskSelection = ({
  config,
  updateConfig,
  onNext,
}: TaskSelectionProps) => {
  const [selected, setSelected] = useState<string | undefined>(config.task);

  const tasks = [
    {
      id: "classification",
      icon: FileText,
      title: "Classification",
      description: "Text classification, sentiment analysis, and entity recognition",
    },
    {
      id: "chatbot",
      icon: MessageSquare,
      title: "Chatbot",
      description: "Conversational AI with FAQ or context-aware responses",
    },
    {
      id: "knowledge",
      icon: BookOpen,
      title: "Knowledge Base",
      description: "RAG-based question answering over documents",
    },
  ];

  const handleSelect = (taskId: string) => {
    setSelected(taskId);
    updateConfig({ task: taskId });
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Select Your Task Type</h2>
        <p className="text-muted-foreground">
          Choose the type of AI model you want to create
        </p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 max-w-4xl mx-auto">
        {tasks.map((task) => (
          <TaskCard
            key={task.id}
            icon={task.icon}
            title={task.title}
            description={task.description}
            selected={selected === task.id}
            onClick={() => handleSelect(task.id)}
          />
        ))}
      </div>

      <div className="flex justify-end">
        <Button
          size="lg"
          onClick={onNext}
          disabled={!selected}
          className="bg-primary hover:bg-primary/90"
        >
          Continue
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

// Model Type Selection Step
const ModelTypeSelection = ({
  config,
  updateConfig,
  onNext,
  onBack,
}: ModelTypeSelectionProps) => {
  const [selected, setSelected] = useState<string | undefined>(config.modelType);

  const modelTypes = [
    {
      id: "auto",
      icon: Sparkles,
      title: "Auto",
      description: "Let AI choose the best model for your task",
    },
    {
      id: "logisticregression",
      icon: Network,
      title: "Logistic Regression",
      description: "Linear classification model for binary and multi-class tasks",
    },
    {
      id: "randomforest",
      icon: Layers,
      title: "Random Forest",
      description: "Ensemble method using multiple decision trees",
    },
    {
      id: "svm",
      icon: Zap,
      title: "SVM",
      description: "Support Vector Machine for classification tasks",
    },
    {
      id: "llm",
      icon: Brain,
      title: "LLM",
      description: "Large Language Model for advanced capabilities",
    },
  ];

  const handleSelect = (typeId: string) => {
    setSelected(typeId);
    updateConfig({ modelType: typeId });
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Choose Your Model Type</h2>
        <p className="text-muted-foreground">
          Select the model algorithm for your task
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6 max-w-7xl mx-auto">
        {modelTypes.map((type) => {
          const Icon = type.icon;
          return (
            <Card
              key={type.id}
              className={cn(
                "p-6 cursor-pointer transition-all duration-300 hover:scale-105 border-2",
                selected === type.id
                  ? "border-primary bg-gradient-to-br from-card to-secondary shadow-[var(--glow-primary)]"
                  : "border-border bg-card hover:border-primary/50"
              )}
              onClick={() => handleSelect(type.id)}
            >
              <div className="flex flex-col items-center text-center space-y-4">
                <div
                  className={cn(
                    "p-4 rounded-full transition-colors",
                    selected === type.id
                      ? "bg-primary text-primary-foreground"
                      : "bg-secondary text-foreground"
                  )}
                >
                  <Icon className="h-8 w-8" />
                </div>
                <div>
                  <h3 className="text-xl font-semibold mb-2">{type.title}</h3>
                  <p className="text-sm text-muted-foreground">{type.description}</p>
                </div>
              </div>
            </Card>
          );
        })}
      </div>

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-2 h-5 w-5" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={onNext}
          disabled={!selected}
          className="bg-primary hover:bg-primary/90"
        >
          Continue
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

/* =========================
   STEP: Dataset Upload (API-connected)
   ========================= */
const DatasetSelectionAPI = ({
  config,
  updateConfig,
  onNext,
  onBack,
}: DatasetSelectionProps) => {
  const [uploadedFile, setUploadedFile] = useState<File | null>(null);
  const { toast } = useToast();
  const [uploading, setUploading] = useState(false);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (!config.task) {
      toast({ title: "Please select a task first", variant: "destructive" });
      return;
    }

    setUploading(true);
    toast({ title: "Uploading dataset...", description: file.name });

    try {
      const res = await api.uploadDataset(config.task, file);
      if (res.file_path) {
        setUploadedFile(file);
        updateConfig({ dataset: res.file_path });
        toast({
          title: "✅ Dataset uploaded",
          description: `Saved as ${res.file_path}`,
        });
      } else {
        toast({ title: "Upload failed", variant: "destructive" });
      }
    } catch (err: any) {
      toast({ title: "Upload failed", description: err?.message || String(err), variant: "destructive" });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Upload Dataset</h2>
        <p className="text-muted-foreground">
          Upload your training data file
        </p>
      </div>

      <div className="grid md:grid-cols-2 gap-6 max-w-3xl mx-auto">
        <Card className="p-8 border-2 border-dashed border-border hover:border-primary/50 transition-colors">
          <label className="cursor-pointer flex flex-col items-center space-y-4">
            <div className="p-4 rounded-full bg-primary/10">
              <Upload className="h-8 w-8 text-primary" />
            </div>
            <div className="text-center">
              <h3 className="text-xl font-semibold mb-2">Upload Dataset</h3>
              <p className="text-sm text-muted-foreground mb-4">
                {config.task === "knowledge" ? "PDF, TXT, or MD format" : "CSV, JSON, or TXT format"}
              </p>
              {uploadedFile && (
                <p className="text-sm text-accent font-medium">
                  ✓ {uploadedFile.name}
                </p>
              )}
            </div>
            <input
              type="file"
              className="hidden"
              accept={config.task === "knowledge" ? ".pdf,.txt,.md" : ".csv,.json,.txt"}
              onChange={handleFileUpload}
            />
          </label>
        </Card>

        <Card className="p-8 border-2 border-border hover:border-primary/50 transition-colors cursor-pointer">
          <div className="flex flex-col items-center space-y-4">
            <div className="p-4 rounded-full bg-accent/10">
              <Database className="h-8 w-8 text-accent" />
            </div>
            <div className="text-center">
              <h3 className="text-xl font-semibold mb-2">Sample Datasets</h3>
              <p className="text-sm text-muted-foreground">
                Use pre-loaded example data
              </p>
            </div>
          </div>
        </Card>
      </div>

      {uploadedFile && (
        <Card className="p-6 max-w-3xl mx-auto bg-secondary border-border">
          <h4 className="font-semibold mb-4">Dataset Preview</h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-muted-foreground">File name:</span>
              <span>{uploadedFile.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Size:</span>
              <span>{(uploadedFile.size / 1024).toFixed(2)} KB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-muted-foreground">Type:</span>
              <span>{uploadedFile.type || "Unknown"}</span>
            </div>
          </div>
        </Card>
      )}

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-2 h-5 w-5" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={onNext}
          disabled={!config.dataset || uploading}
          className="bg-primary hover:bg-primary/90"
        >
          Continue
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

/* =========================
   Preprocessing Config Step (unchanged)
   ========================= */
const PreprocessingConfig = ({
  config,
  updateConfig,
  onNext,
  onBack,
}: PreprocessingConfigProps) => {
  const [textColumn, setTextColumn] = useState(
    config.preprocessing?.textColumn || "text"
  );
  const [labelColumn, setLabelColumn] = useState(
    config.preprocessing?.labelColumn || "label"
  );
  const [trainSplit, setTrainSplit] = useState(
    config.preprocessing?.trainSplit || 80
  );

  const handleContinue = () => {
    updateConfig({
      preprocessing: {
        textColumn,
        labelColumn,
        trainSplit,
      },
    });
    onNext();
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Preprocess Settings</h2>
        <p className="text-muted-foreground">
          Configure preprocessing parameters
        </p>
      </div>

      <Card className="p-8 max-w-2xl mx-auto bg-card border-border">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="text-column">Text Column Name</Label>
            <Input
              id="text-column"
              value={textColumn}
              onChange={(e) => setTextColumn(e.target.value)}
              placeholder="e.g., text, content, message"
              className="bg-secondary border-border"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="label-column">Label Column Name</Label>
            <Input
              id="label-column"
              value={labelColumn}
              onChange={(e) => setLabelColumn(e.target.value)}
              placeholder="e.g., label, category, sentiment"
              className="bg-secondary border-border"
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Test Split</Label>
              <span className="text-sm text-muted-foreground">
                {(100 - trainSplit) / 100} ({(100 - trainSplit)}%)
              </span>
            </div>
            <Slider
              value={[trainSplit]}
              onValueChange={(value) => setTrainSplit(value[0])}
              min={50}
              max={95}
              step={5}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-muted-foreground">
              <span>50% Train</span>
              <span>95% Train</span>
            </div>
          </div>
        </div>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-2 h-5 w-5" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={handleContinue}
          className="bg-primary hover:bg-primary/90"
        >
          Continue
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

/* =========================
   Training Config Step (unchanged)
   ========================= */
const TrainingConfig = ({
  config,
  updateConfig,
  onNext,
  onBack,
}: TrainingConfigProps) => {
  const [modelName, setModelName] = useState(
    config.training?.modelName || "my-model"
  );
  const [budget, setBudget] = useState(config.training?.budget || 5);
  const [asyncTraining, setAsyncTraining] = useState(
    config.training?.async ?? true
  );

  const handleStartTraining = () => {
    updateConfig({
      training: {
        modelName,
        budget,
        async: asyncTraining,
      },
    });
    onNext();
  };

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Start Training</h2>
        <p className="text-muted-foreground">
          Configure training parameters before starting
        </p>
      </div>

      <Card className="p-8 max-w-2xl mx-auto bg-card border-border">
        <div className="space-y-6">
          <div className="space-y-2">
            <Label htmlFor="model-name">Model Name</Label>
            <Input
              id="model-name"
              value={modelName}
              onChange={(e) => setModelName(e.target.value)}
              placeholder="e.g., sentiment-analyzer-v1"
              className="bg-secondary border-border"
            />
          </div>

          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <Label>Epochs</Label>
              <span className="text-sm text-muted-foreground">{budget}</span>
            </div>
            <Slider
              value={[budget]}
              onValueChange={(value) => setBudget(value[0])}
              min={1}
              max={60}
              step={1}
              className="w-full"
            />
            <p className="text-xs text-muted-foreground">
              Number of training epochs (iterations over the dataset)
            </p>
          </div>

          <div className="flex items-center justify-between p-4 rounded-lg bg-secondary">
            <div className="space-y-1">
              <Label htmlFor="async-mode">Background Training</Label>
              <p className="text-sm text-muted-foreground">
                Train in background and get notified when complete
              </p>
            </div>
            <Switch
              id="async-mode"
              checked={asyncTraining}
              onCheckedChange={setAsyncTraining}
            />
          </div>
        </div>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack}>
          <ArrowLeft className="mr-2 h-5 w-5" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={handleStartTraining}
          className="bg-primary hover:bg-primary/90"
        >
          Start Training
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

/* =========================
   Model Training (API-polling)
   ========================= */
const ModelTrainingAPI = ({ config, onNext, onBack, updateConfig }: ModelTrainingProps) => {
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState("Initializing...");
  const [isComplete, setIsComplete] = useState(false);
  const { toast } = useToast();

  useEffect(() => {
    let intervalId: number | undefined;

    const startTraining = async () => {
      if (!config.task || !config.dataset) {
        toast({ title: "Missing task or dataset", variant: "destructive" });
        return;
      }

      setStatus("Starting training job...");
      try {
        const res = await api.startTraining({
          taskType: config.task,
          modelType: (config.modelType as string) || "auto",
          datasetPath: (config.dataset as string) || "",
          testSplit: ((config.preprocessing?.trainSplit ?? config.preprocessing?.trainSplit) ? ((config.preprocessing?.trainSplit || 80) / 100) : 0.2) as number,
          epochs: config.training?.budget || 1,
        });

        if (!res.job_id) {
          setStatus("Failed to start training");
          toast({ title: "Training failed to start", variant: "destructive" });
          return;
        }

        const jobId = res.job_id;
        setStatus("Training in progress...");

        intervalId = window.setInterval(async () => {
          try {
            const statusRes = await api.checkStatus(jobId);
            // assume statusRes.status: "queued" | "running" | "completed" | "failed"
            if (statusRes.status === "completed" || statusRes.status === "finished") {
              if (intervalId) window.clearInterval(intervalId);
              setProgress(100);
              setStatus("✅ Training complete!");
              setIsComplete(true);
              // attach model info into config
              updateConfig({ modelInfo: statusRes.result || statusRes.result || statusRes });
              toast({ title: "Training completed", description: "Model ready" });
            } else if (statusRes.status === "failed" || statusRes.status === "error") {
              if (intervalId) window.clearInterval(intervalId);
              setStatus("❌ Training failed");
              toast({ title: "Training failed", description: statusRes.error || "See server logs", variant: "destructive" });
            } else {
              // update progress conservatively (avoid direct dependency on closure)
              setProgress((p) => Math.min(90, p + 8));
              setStatus(statusRes.status || "Running...");
            }
          } catch (err: any) {
            // polling error; show lightweight warning but keep trying
            setStatus("Polling backend...");
          }
        }, 3000);
      } catch (err: any) {
        setStatus("Failed to start training");
        toast({ title: "Training start error", description: err?.message || String(err), variant: "destructive" });
      }
    };

    startTraining();

    return () => {
      if (intervalId) window.clearInterval(intervalId);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // run once on mount

  return (
    <div className="space-y-8 animate-in fade-in duration-500">
      <div className="text-center space-y-2">
        <h2 className="text-3xl font-bold">Training in Progress</h2>
        <p className="text-muted-foreground">
          Please wait while we train your model
        </p>
      </div>

      <Card className="p-8 max-w-2xl mx-auto bg-card border-border">
        <div className="space-y-6">
          <div className="flex items-center justify-center">
            {isComplete ? (
              <div className="flex items-center space-x-2 text-accent">
                <CheckCircle className="h-8 w-8" />
                <span className="text-xl font-semibold">Training Complete!</span>
              </div>
            ) : (
              <div className="animate-pulse flex items-center space-x-2">
                <div className="h-3 w-3 bg-primary rounded-full animate-bounce" />
                <div
                  className="h-3 w-3 bg-primary rounded-full animate-bounce"
                  style={{ animationDelay: "0.2s" }}
                />
                <div
                  className="h-3 w-3 bg-primary rounded-full animate-bounce"
                  style={{ animationDelay: "0.4s" }}
                />
              </div>
            )}
          </div>

          <div className="space-y-2">
            <Progress value={progress} className="h-3" />
            <p className="text-center text-sm text-muted-foreground">{status}</p>
          </div>

          <div className="grid grid-cols-2 gap-4 p-4 rounded-lg bg-secondary">
            <div>
              <p className="text-sm text-muted-foreground">Model Name</p>
              <p className="font-medium">{config.training?.modelName}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Epochs</p>
              <p className="font-medium">{config.training?.budget}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Task Type</p>
              <p className="font-medium capitalize">{config.task}</p>
            </div>
            <div>
              <p className="text-sm text-muted-foreground">Model Type</p>
              <p className="font-medium capitalize">{config.modelType}</p>
            </div>
          </div>
        </div>
      </Card>

      <div className="flex justify-between">
        <Button variant="outline" size="lg" onClick={onBack} disabled={!isComplete}>
          <ArrowLeft className="mr-2 h-5 w-5" />
          Back
        </Button>
        <Button
          size="lg"
          onClick={onNext}
          disabled={!isComplete}
          className="bg-primary hover:bg-primary/90"
        >
          Test Model
          <ArrowRight className="ml-2 h-5 w-5" />
        </Button>
      </div>
    </div>
  );
};

/* =========================
   Model Testing (download/export) + Main wiring (updated)
   ========================= */

const ModelTestingAPI = ({ config, onBack }: { config: ModelConfig; onBack: () => void }) => {
  const [input, setInput] = useState("");
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const { toast } = useToast();

  const handleTest = async () => {
    try {
      if (!input.trim()) {
        toast({ title: "Enter sample text to test", variant: "destructive" });
        return;
      }

      // Prefer explicit model_path, otherwise try to derive from zip_path
      const modelPath =
        (config.modelInfo && (config.modelInfo.model_path || config.modelInfo.model_path === "")
          ? config.modelInfo.model_path
          : undefined) ||
        (config.modelInfo?.zip_path ? (config.modelInfo.zip_path as string).replace(/\.zip$/i, ".pkl") : undefined);

      if (!modelPath) {
        toast({
          title: "Model not available",
          description: "Train a model first or check the training result",
          variant: "destructive",
        });
        return;
      }

      setLoading(true);

      // 1) Load model into backend cache
      await api.loadModel(modelPath);

      // 2) Run prediction
      const res = await api.predictText(input);

      // backend returns { ok: true, result: ... } or { ok: false, error: ... }
      if (res == null) throw new Error("Empty response from server");
      if (res.ok === false) {
        throw new Error(res.error || JSON.stringify(res));
      }

      // Pretty-print the returned result if present
      const pretty = res.result !== undefined ? JSON.stringify(res.result, null, 2) : JSON.stringify(res, null, 2);
      setResult(pretty);
      toast({ title: "Prediction successful" });
    } catch (err: any) {
      toast({
        title: "Prediction failed",
        description: err?.message || String(err),
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleExport = () => {
    if (!config.modelInfo?.zip_path) {
      toast({ title: "No model found", description: "Train the model first", variant: "destructive" });
      return;
    }

    // zip_path usually like: C:\...\outputs\packages\name.zip or /.../name.zip
    const zipPath = config.modelInfo.zip_path as string;
    const filename = zipPath.split(/[\\/]/).pop();
    if (!filename) {
      toast({ title: "Download failed", variant: "destructive" });
      return;
    }
    const url = api.getDownloadUrl(filename);
    window.open(url, "_blank");
    toast({ title: "Downloading model...", description: filename });
  };

  // optional: show plot if available (assumes backend makes it available via download endpoint)
  const plotUrl =
    config.modelInfo?.plot_path
      ? api.getDownloadUrl((config.modelInfo.plot_path as string).split(/[\\/]/).pop() || "")
      : null;

  return (
    <div className="space-y-6 text-center">
      <h2 className="text-2xl font-bold mb-4">Test & Export Model</h2>

      <Card className="p-8">
        <Textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter sample text to test..."
          className="min-h-[120px] mb-4"
        />
        <div className="flex items-center justify-center gap-3">
          <Button onClick={handleTest} disabled={!input || loading}>
            {loading ? "Running..." : "Run Test"}
          </Button>
          <Button variant="ghost" onClick={() => { setInput(""); setResult(null); }}>
            Clear
          </Button>
        </div>

        {result && (
          <pre className="mt-4 text-accent font-mono text-sm whitespace-pre-wrap bg-secondary p-4 rounded-lg text-left overflow-auto">
            {result}
          </pre>
        )}

        {plotUrl && (
          <div className="mt-6">
            <img src={plotUrl} alt="Model performance" className="mx-auto rounded-lg border border-border max-w-full" />
          </div>
        )}
      </Card>

      <div className="flex justify-center gap-4">
        <Button variant="outline" onClick={onBack}>
          <ArrowLeft className="mr-2 h-5 w-5" /> Back
        </Button>
        <Button onClick={handleExport}>
          <Download className="mr-2 h-5 w-5" /> Export Model
        </Button>
        <Button onClick={() => toast({ title: "Deploy feature coming soon" })}>
          <Rocket className="mr-2 h-5 w-5" /> Deploy
        </Button>
      </div>
    </div>
  );
};

/* =========================
   Main CreateModel Component wiring
   ========================= */
const getSteps = (task?: string) => {
  const baseSteps = [
    { id: 1, name: "Task", description: "Select your task type" },
    { id: 2, name: "Model", description: "Choose your model type" },
    { id: 3, name: "Dataset", description: "Upload dataset" },
  ];
  
  if (task === "knowledge") {
    return [
      ...baseSteps,
      { id: 5, name: "Training", description: "Start training" },
      { id: 6, name: "Train", description: "Training in progress" },
      { id: 7, name: "Test", description: "Test and deploy" },
    ];
  }
  
  return [
    ...baseSteps,
    { id: 4, name: "Preprocess", description: "Preprocess settings" },
    { id: 5, name: "Training", description: "Start training" },
    { id: 6, name: "Train", description: "Training in progress" },
    { id: 7, name: "Test", description: "Test and deploy" },
  ];
};

const CreateModel = () => {
  const [currentStep, setCurrentStep] = useState(1);
  const [config, setConfig] = useState<ModelConfig>({});

  const updateConfig = (updates: Partial<ModelConfig>) => {
    setConfig((prev) => ({ ...prev, ...updates }));
  };

  const nextStep = () => {
    const steps = getSteps(config.task);
    setCurrentStep((prev) => Math.min(prev + 1, steps.length));
  };

  const prevStep = () => {
    setCurrentStep((prev) => Math.max(prev - 1, 1));
  };

  const steps = getSteps(config.task);

  return (
    <div className="min-h-screen py-12">
      <div className="container mx-auto px-4 max-w-6xl">
        <div className="mb-16">
          <StepIndicator steps={steps} currentStep={currentStep} />
        </div>

        <div className="min-h-[500px]">
          {currentStep === 1 && (
            <TaskSelection
              config={config}
              updateConfig={updateConfig}
              onNext={nextStep}
            />
          )}
          {currentStep === 2 && (
            <ModelTypeSelection
              config={config}
              updateConfig={updateConfig}
              onNext={nextStep}
              onBack={prevStep}
            />
          )}
          {currentStep === 3 && (
            // use API-connected upload step
            <DatasetSelectionAPI
              config={config}
              updateConfig={updateConfig}
              onNext={() => {
                // Skip preprocessing for knowledge base
                if (config.task === "knowledge") {
                  setCurrentStep(5);
                } else {
                  setCurrentStep(4);
                }
              }}
              onBack={prevStep}
            />
          )}
          {currentStep === 4 && config.task !== "knowledge" && (
            <PreprocessingConfig
              config={config}
              updateConfig={updateConfig}
              onNext={() => setCurrentStep(5)}
              onBack={() => setCurrentStep(3)}
            />
          )}
          {currentStep === 5 && (
            <TrainingConfig
              config={config}
              updateConfig={updateConfig}
              onNext={() => setCurrentStep(6)}
              onBack={() => config.task === "knowledge" ? setCurrentStep(3) : setCurrentStep(4)}
            />
          )}
          {currentStep === 6 && (
            <ModelTrainingAPI
              config={config}
              updateConfig={updateConfig}
              onNext={() => setCurrentStep(7)}
              onBack={() => setCurrentStep(5)}
            />
          )}
          {currentStep === 7 && (
            <ModelTestingAPI
              config={config}
              onBack={() => setCurrentStep(6)}
            />
          )}
        </div>
      </div>
    </div>
  );
};

export default CreateModel;
